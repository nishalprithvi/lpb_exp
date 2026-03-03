from typing import Optional, Callable, Generator
import numpy as np
import imageio

try:
    import av  # type: ignore
except ImportError:
    av = None

from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs


def read_video(
        video_path: str, dt: float,
        video_start_time: float=0.0,
        start_time: float=0.0,
        img_transform: Optional[Callable[[np.ndarray], np.ndarray]]=None,
        thread_type: str="AUTO",
        thread_count: int=0,
        max_pad_frames: int=10
        ) -> Generator[np.ndarray, None, None]:
    frame = None
    next_global_idx = 0

    if av is not None:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = thread_type
            stream.thread_count = thread_count
            for _, frame in enumerate(container.decode(stream)):
                since_start = frame.time
                frame_time = video_start_time + since_start
                _, global_idxs, next_global_idx = get_accumulate_timestamp_idxs(
                    timestamps=[frame_time],
                    start_time=start_time,
                    dt=dt,
                    next_global_idx=next_global_idx,
                )
                if len(global_idxs) > 0:
                    array = frame.to_ndarray(format="rgb24")
                    img = img_transform(array) if img_transform is not None else array
                    for _ in global_idxs:
                        yield img
        if frame is not None:
            array = frame.to_ndarray(format="rgb24")
            img = img_transform(array) if img_transform is not None else array
            for _ in range(max_pad_frames):
                yield img
        return

    # imageio fallback when pyav is unavailable.
    with imageio.get_reader(video_path) as reader:
        fps_meta = reader.get_meta_data().get("fps", None)
        fps = float(fps_meta) if fps_meta else (1.0 / dt)
        for frame_idx, array in enumerate(reader):
            frame_time = video_start_time + (frame_idx / fps)
            _, global_idxs, next_global_idx = get_accumulate_timestamp_idxs(
                timestamps=[frame_time],
                start_time=start_time,
                dt=dt,
                next_global_idx=next_global_idx,
            )
            if len(global_idxs) > 0:
                img = img_transform(array) if img_transform is not None else array
                for _ in global_idxs:
                    yield img
            frame = array

    if frame is not None:
        img = img_transform(frame) if img_transform is not None else frame
        for _ in range(max_pad_frames):
            yield img


class VideoRecorder:
    def __init__(self, fps, codec, input_pix_fmt, **kwargs):
        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        self._reset_state()

    def _reset_state(self):
        self.container = None
        self.stream = None
        self.writer = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0

    @classmethod
    def create_h264(cls,
            fps,
            codec="h264",
            input_pix_fmt="rgb24",
            output_pix_fmt="yuv420p",
            crf=18,
            profile="high",
            **kwargs):
        return cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={"crf": str(crf), "profile": profile},
            **kwargs,
        )

    def __del__(self):
        self.stop()

    def is_ready(self):
        return (self.stream is not None) or (self.writer is not None)

    def start(self, file_path, start_time=None):
        if self.is_ready():
            self.stop()

        if av is not None:
            self.container = av.open(file_path, mode="w")
            self.stream = self.container.add_stream(self.codec, rate=self.fps)
            codec_context = self.stream.codec_context
            for k, v in self.kwargs.items():
                setattr(codec_context, k, v)
        else:
            self.writer = imageio.get_writer(file_path, fps=self.fps)

        self.start_time = start_time

    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError("Must run start() before writing!")

        n_repeats = 1
        if self.start_time is not None:
            local_idxs, _, self.next_global_idx = get_accumulate_timestamp_idxs(
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1 / self.fps,
                next_global_idx=self.next_global_idx,
            )
            n_repeats = len(local_idxs)

        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            if self.stream is not None:
                h, w, _ = img.shape
                self.stream.width = w
                self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype

        if self.stream is not None:
            frame = av.VideoFrame.from_ndarray(img, format=self.input_pix_fmt)
            for _ in range(n_repeats):
                for packet in self.stream.encode(frame):
                    self.container.mux(packet)
        else:
            for _ in range(n_repeats):
                self.writer.append_data(img)

    def stop(self):
        if not self.is_ready():
            return

        if self.stream is not None:
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
        if self.writer is not None:
            self.writer.close()

        self._reset_state()
