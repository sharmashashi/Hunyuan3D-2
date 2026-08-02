# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import ctypes
import gc
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import traceback
import uuid
from io import BytesIO

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer, \
    MeshSimplifier
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline

LOGDIR = '.'

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


class ModelWorker:
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2mini',
                 tex_model_path='tencent/Hunyuan3D-2',
                 subfolder='hunyuan3d-dit-v2-mini-turbo',
                 device='cuda',
                 enable_tex=False,
                 low_vram_mode=False):
        # NOTE: low_vram_mode is accepted for CLI/script compatibility. Since
        # models are now unloaded to disk after every job there is no resident
        # VRAM/RAM footprint left to manage between requests.
        self.model_path = model_path
        self.tex_model_path = tex_model_path
        self.subfolder = subfolder
        self.worker_id = worker_id
        self.device = device
        self.enable_tex = enable_tex
        # Serialize jobs: pipelines are created/destroyed per request, so
        # concurrent requests must not race on the model lifecycle.
        self._lock = threading.Lock()
        logger.info(f"Creating worker {worker_id} (shape={model_path}/{subfolder}, tex={enable_tex}) ...")

        # Small ONNX model used for background removal; stays resident.
        self.rembg = BackgroundRemover()

        # Pipelines are loaded lazily from disk per request and released
        # (GPU + RAM) as soon as the request finishes. Only one pipeline is
        # ever resident at a time, so both VRAM and system RAM return to
        # near-idle between requests and full-size models become runnable on
        # a 12 GB card that could never hold shape + texture simultaneously.
        self.pipeline = None
        self.pipeline_tex = None
        torch.cuda.empty_cache()

    @staticmethod
    def _free_memory():
        """GC + release cached CUDA blocks + return freed heap pages to the OS."""
        gc.collect()
        torch.cuda.empty_cache()
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass

    # -- shape pipeline ---------------------------------------------------------
    def _load_shape(self):
        if self.pipeline is None:
            logger.info("Loading shape pipeline from disk ...")
            self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                self.model_path,
                subfolder=self.subfolder,
                use_safetensors=True,
                device=self.device,
            )
            self.pipeline.enable_flashvdm(mc_algo='mc')
            torch.cuda.empty_cache()
        return self.pipeline

    def _release_shape(self):
        if self.pipeline is not None:
            logger.info("Unloading shape pipeline from GPU and RAM ...")
            del self.pipeline
            self.pipeline = None
            self._free_memory()

    # -- texture pipeline -------------------------------------------------------
    def _load_tex(self):
        if self.pipeline_tex is None:
            logger.info("Loading texture pipeline from disk ...")
            self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(self.tex_model_path)
            # Texture models stream GPU<->CPU during generation; once the job
            # finishes the whole pipeline is released so nothing stays in RAM.
            logger.info("Enabling model CPU offload for texture generation")
            self.pipeline_tex.enable_model_cpu_offload()
            torch.cuda.empty_cache()
        return self.pipeline_tex

    def _release_tex(self):
        if self.pipeline_tex is not None:
            logger.info("Unloading texture pipeline from GPU and RAM ...")
            del self.pipeline_tex
            self.pipeline_tex = None
            self._free_memory()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        # Pipelines are created and destroyed per request, so serialize jobs.
        with self._lock:
            return self._generate_locked(uid, params)

    def _generate_locked(self, uid, params):
        if 'image' in params:
            image = params["image"]
            image = load_image_from_base64(image)
        else:
            if 'text' in params:
                text = params["text"]
                image = self.pipeline_t2i(text)
            else:
                raise ValueError("No input image or text provided")

        image = self.rembg(image)
        params['image'] = image

        if 'mesh' in params:
            mesh = trimesh.load(BytesIO(base64.b64decode(params["mesh"])), file_type='glb')
        else:
            self._load_shape()
            try:
                seed = params.get("seed", 1234)
                params['generator'] = torch.Generator(self.device).manual_seed(seed)
                params['octree_resolution'] = params.get("octree_resolution", 128)
                params['num_inference_steps'] = params.get("num_inference_steps", 5)
                params['guidance_scale'] = params.get('guidance_scale', 5.0)
                params['mc_algo'] = 'mc'
                import time
                start_time = time.time()
                mesh = self.pipeline(**params)[0]
                logger.info("--- shape generation %s seconds ---" % (time.time() - start_time))
            finally:
                # Shape is not needed for texture generation and must not stay
                # resident between jobs: free it (GPU + RAM) right away.
                self._release_shape()

        if params.get('texture', False):
            self._release_shape()
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=params.get('face_count', 40000))
            try:
                self._load_tex()
                import time
                start_time = time.time()
                mesh = self.pipeline_tex(mesh, image)
                logger.info("--- texture generation %s seconds ---" % (time.time() - start_time))
            finally:
                self._release_tex()

        type = params.get('type', 'glb')
        with tempfile.NamedTemporaryFile(suffix=f'.{type}', delete=False) as temp_file:
            mesh.export(temp_file.name)
            mesh = trimesh.load(temp_file.name)
            save_path = os.path.join(SAVE_DIR, f'{str(uid)}.{type}')
            mesh.export(save_path)

        self._free_memory()
        return save_path, uid


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 你可以指定允许的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)


@app.post("/generate")
async def generate(request: Request):
    logger.info("Worker generating...")
    params = await request.json()
    uid = uuid.uuid4()
    try:
        file_path, uid = worker.generate(uid, params)
        return FileResponse(file_path)
    except ValueError as e:
        traceback.print_exc()
        print("Caught ValueError:", e)
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except torch.cuda.CudaError as e:
        print("Caught torch.cuda.CudaError:", e)
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except Exception as e:
        print("Caught Unknown Error", e)
        traceback.print_exc()
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)


@app.post("/send")
async def generate(request: Request):
    logger.info("Worker send...")
    params = await request.json()
    uid = uuid.uuid4()
    threading.Thread(target=worker.generate, args=(uid, params,)).start()
    ret = {"uid": str(uid)}
    return JSONResponse(ret, status_code=200)


@app.get("/status/{uid}")
async def status(uid: str):
    save_file_path = os.path.join(SAVE_DIR, f'{uid}.glb')
    print(save_file_path, os.path.exists(save_file_path))
    if not os.path.exists(save_file_path):
        response = {'status': 'processing'}
        return JSONResponse(response, status_code=200)
    else:
        base64_str = base64.b64encode(open(save_file_path, 'rb').read()).decode()
        response = {'status': 'completed', 'model_base64': base64_str}
        return JSONResponse(response, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--tex_model_path", type=str, default='tencent/Hunyuan3D-2')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument('--enable_tex', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true')
    parser.add_argument('--subfolder', type=str, default='hunyuan3d-dit-v2-mini-turbo')
    args = parser.parse_args()
    logger.info(f"args: {args}")

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    worker = ModelWorker(model_path=args.model_path, device=args.device, enable_tex=args.enable_tex,
                         tex_model_path=args.tex_model_path, subfolder=args.subfolder,
                         low_vram_mode=args.low_vram_mode)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
