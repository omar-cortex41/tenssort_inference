import time

class FrameSynchronizer:
    """
    Synchronizes frames from multiple RTSP streams using a 'soft barrier' approach.
    
    Ensures that frames are retrieved in a time-aligned manner across all active cameras.
    If a camera lags behind, the synchronizer waits (up to a timeout) to maintain alignment,
    preventing faster streams from drifting ahead of slower ones.
    """
    def __init__(self, provider, stream_ids=None):
        self.provider = provider
        self.stream_ids = list(stream_ids if stream_ids is not None else range(provider.stream_count()))
        self.last_frame_ids = {i: -1 for i in self.stream_ids}

    def get_batch(self, timeout_sec=0.04):
        start_time = time.time()
        batch_result = {cam_id: None for cam_id in self.stream_ids}
        pending_cams = set(self.stream_ids)

        while pending_cams:
            if time.time() - start_time > timeout_sec:
                break 

            for cam_id in list(pending_cams):
                info = self.provider.get_cuda_frame(cam_id)
                
                if info.get("valid") and info['frame_id'] > self.last_frame_ids[cam_id]:
                    batch_result[cam_id] = info
                    self.last_frame_ids[cam_id] = info['frame_id']
                    pending_cams.remove(cam_id)
            
            if pending_cams:
                time.sleep(0.001)

        return batch_result