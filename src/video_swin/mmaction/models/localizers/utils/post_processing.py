from video_swin.mmaction.localization import soft_nms


def post_processing(result, video_info, soft_nms_alpha, soft_nms_low_threshold,
                    soft_nms_high_threshold, post_process_top_k,
                    feature_extraction_interval):
    """Post process for temporal proposals generation.

    Args:
        result (np.ndarray): Proposals generated by network.
        video_info (dict): Meta data of video. Required keys are
            'duration_frame', 'duration_second'.
        soft_nms_alpha (float): Alpha value of Gaussian decaying function.
        soft_nms_low_threshold (float): Low threshold for soft nms.
        soft_nms_high_threshold (float): High threshold for soft nms.
        post_process_top_k (int): Top k values to be considered.
        feature_extraction_interval (int): Interval used in feature extraction.

    Returns:
        list[dict]: The updated proposals, e.g.
            [{'score': 0.9, 'segment': [0, 1]},
             {'score': 0.8, 'segment': [0, 2]},
            ...].
    """
    if len(result) > 1:
        result = soft_nms(result, soft_nms_alpha, soft_nms_low_threshold,
                          soft_nms_high_threshold, post_process_top_k)

    result = result[result[:, -1].argsort()[::-1]]
    video_duration = float(
        video_info['duration_frame'] // feature_extraction_interval *
        feature_extraction_interval
    ) / video_info['duration_frame'] * video_info['duration_second']
    proposal_list = []

    for j in range(min(post_process_top_k, len(result))):
        proposal = {}
        proposal['score'] = float(result[j, -1])
        proposal['segment'] = [
            max(0, result[j, 0]) * video_duration,
            min(1, result[j, 1]) * video_duration
        ]
        proposal_list.append(proposal)
    return proposal_list