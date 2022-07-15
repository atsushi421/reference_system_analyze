import pandas as pd

from caret_analyze import Architecture, Lttng, Application
from latency_dataframe_getter import LatencyDataFrameGetter


class AllLatencyGetter:
    def __init__(
        self,
        arch_path: str,
        ctf_path: str
    ) -> None:
        arch = Architecture('yaml', arch_path)
        lttng = Lttng(ctf_path)
        app = Application(arch, lttng)
        self._getter = LatencyDataFrameGetter(app)

    def get_all_nodes_latency_df(
        self
    ) -> pd.DataFrame:
        main_df = self._getter.get_node_latency_df('main_path')

        sub_df = self._getter.get_node_latency_df('sub_path')
        sub_df.drop(columns=['/PointCloudFusion', '/BehaviorPlanner'], inplace=True)

        point_cloud_map_df = self._getter.get_node_latency_df('point_cloud_map')
        point_cloud_map_df.drop(columns=['/NDTLocalizer', '/BehaviorPlanner'], inplace=True)

        visualizer_df = self._getter.get_node_latency_df('visualizer')
        visualizer_df.drop(columns=['/Lanelet2GlobalPlanner', '/BehaviorPlanner'], inplace=True)

        lanelet2map_df = self._getter.get_node_latency_df('lanelet2map')
        lanelet2map_df.drop(columns=['/Lanelet2MapLoader', '/BehaviorPlanner'], inplace=True)

        euclidean_cluster_settings_df = self._getter.get_node_latency_df('euclidean_cluster_settings')
        euclidean_cluster_settings_df.drop(columns=['/EuclideanClusterDetector'], inplace=True)

        all_nodes_latency_df = pd.concat(
            [
                main_df,
                sub_df,
                point_cloud_map_df,
                visualizer_df,
                lanelet2map_df,
                euclidean_cluster_settings_df
            ],
            axis=1
        )

        return all_nodes_latency_df
    
    def get_all_edges_latency_df(
        self
    ) -> pd.DataFrame:
        main_df = self._getter.get_edge_latency_df('main_path')
        sub_df = self._getter.get_edge_latency_df('sub_path')
        point_cloud_map_df = self._getter.get_edge_latency_df('point_cloud_map')
        visualizer_df = self._getter.get_edge_latency_df('visualizer')
        lanelet2map_df = self._getter.get_edge_latency_df('lanelet2map')
        euclidean_cluster_settings_df = self._getter.get_edge_latency_df('euclidean_cluster_settings')

        all_edges_latency_df = pd.concat(
            [
                main_df,
                sub_df,
                point_cloud_map_df,
                visualizer_df,
                lanelet2map_df,
                euclidean_cluster_settings_df
            ],
            axis=1
        )

        return all_edges_latency_df
