import logging

import pytrec_eval

logger = logging.getLogger(__name__)


def log(msg: str) -> None:
    logger.info(msg)


def evaluate_retrieval(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
    ignore_identical_ids: bool = True,
    round_to: int = 5,
) -> dict[str, float]:
    """Adapted from: https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/evaluation.py"""
    if ignore_identical_ids:
        logger.info(
            "For evaluation, we ignore identical query and document ids (default), "
            "please explicitly set ``ignore_identical_ids=False`` to ignore this."
        )
        popped = []
        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)
                    popped.append(pid)

    res: dict[str, float] = {}

    for k in k_values:
        res[f"NDCG@{k}"] = 0.0
        res[f"MAP@{k}"] = 0.0
        res[f"R@{k}"] = 0.0
        res[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            res[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            res[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            res[f"R@{k}"] += scores[query_id]["recall_" + str(k)]
            res[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        res[f"NDCG@{k}"] = round(res[f"NDCG@{k}"] / len(scores), round_to)
        res[f"MAP@{k}"] = round(res[f"MAP@{k}"] / len(scores), round_to)
        res[f"R@{k}"] = round(res[f"R@{k}"] / len(scores), round_to)
        res[f"P@{k}"] = round(res[f"P@{k}"] / len(scores), round_to)

    return res
