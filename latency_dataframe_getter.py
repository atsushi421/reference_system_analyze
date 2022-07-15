import pandas as pd

from caret_analyze import Application
from caret_analyze.runtime import CallbackBase, Node


class LatencyDataFrameGetter:
    def __init__(
        self,
        app: Application
    ) -> None:
        self._app = app

    def get_node_latency_df(
        self,
        target_path_name: str
    ) -> pd.DataFrame:
        # Initialize variables
        path = self._app.get_path(target_path_name)
        node_latency_df = pd.DataFrame(
            columns=[node_path.node_name for node_path in path.node_paths])

        # Get head node latency
        head_node_name = path.summary['path'][0]['node']
        head_node = self._app.get_node(head_node_name)
        node_latency_df[head_node_name] = self._get_node_latency(head_node)

        for node_path in path.node_paths:
            if node_path.column_names:
                _, latency = node_path.to_timeseries(
                    remove_dropped=True,
                    treat_drop_as_delay=False,
                    lstrip_s=0,
                    rstrip_s=0,
                )
                node_latency_df[node_path.node_name] = \
                    pd.Series(latency * 1.0e-6)

        # Get tail node latency
        tail_node_name = path.summary['path'][-1]['node']
        tail_node = self._app.get_node(tail_node_name)
        node_latency_df[tail_node_name] = self._get_node_latency(tail_node)

        return node_latency_df

    def _get_node_latency(
        self,
        node: Node
    ) -> pd.Series:
        '''HACK'''
        callbacks = node.get_callbacks(*node.callback_names)
        node_latency = self._get_callback_latency(callbacks[0])

        if len(callbacks) >= 2:
            for callback in callbacks[1:]:
                node_latency += self._get_callback_latency(callback)
        
        return node_latency

    def _get_callback_latency(
        self,
        callback: CallbackBase
    ) -> pd.Series:
        cb_latency_list = []
        cb_latency_df = callback.to_dataframe()
        for ts_tuple in cb_latency_df.itertuples():
            latency = (ts_tuple._2 - ts_tuple._1) * 1.0e-6
            cb_latency_list.append(latency)

        return pd.Series(cb_latency_list)
        
