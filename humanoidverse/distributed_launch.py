import os


def isolate_local_rank_visible_device() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 1:
        return
    if os.environ.get("BFM_DISABLE_CUDA_VISIBLE_ISOLATION", "0") == "1":
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if len(devices) <= 1:
            os.environ["BFM_CUDA_VISIBLE_ISOLATED"] = "1"
            return
        if local_rank >= len(devices):
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but CUDA_VISIBLE_DEVICES only has {len(devices)} entries: {visible}"
            )
        os.environ["BFM_ORIGINAL_CUDA_VISIBLE_DEVICES"] = visible
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["BFM_CUDA_VISIBLE_ISOLATED"] = "1"
