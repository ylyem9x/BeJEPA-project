import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mediapy
import numpy as np
import pandas as pd


def logfile_to_video_directory(logfile: str | Path):
    log_dir = Path(logfile).parent
    video_dir = log_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    return video_dir


@dataclasses.dataclass
class CSVLogger:
    filename: Union[str, Path]
    fields: Optional[List[str]] = None

    def log(self, log_data: Dict[str, Any]) -> None:
        if self.fields is None:
            self.fields = sorted(list(log_data.keys()))
            if not Path(self.filename).exists():
                pd.DataFrame(columns=self.fields).to_csv(self.filename, index=False)
        data = {field: log_data.get(field, "") for field in self.fields}
        islist = [isinstance(v, Iterable) and (not isinstance(v, str)) for (k, v) in data.items()]
        if all(islist):
            df = pd.DataFrame(data)
        elif not any(islist):
            df = pd.DataFrame([data])
        else:
            raise RuntimeError(
                "Fields should all be a numbers, a string or iterable objects. We don't support mixed types."
            )
        df.to_csv(self.filename, mode="a", header=False, index=False)

    def log_video(self, filename: str, frames: list[np.ndarray], fps: int) -> None:
        output_path = logfile_to_video_directory(self.filename) / filename
        mediapy.write_video(output_path, frames, fps=fps)


@dataclasses.dataclass
class JSONLogger:
    filename: Union[str, Path]
    fields: Optional[List[str]] = None

    def log(self, log_data: Dict[str, Any]) -> None:
        if self.fields is None:
            self.fields = sorted(list(log_data.keys()))
            if not Path(self.filename).exists():
                with open(self.filename, "w+") as f:
                    json.dump({k: [] for k in self.fields}, f)
        with open(self.filename, "r+") as f:
            logz = json.load(f)
        with open(self.filename, "w+") as f:
            for field in self.fields:
                logz[field].append(log_data.get(field, ""))
            json.dump(logz, f)
