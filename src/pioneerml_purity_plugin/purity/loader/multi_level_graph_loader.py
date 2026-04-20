from __future__ import annotations

import torch

from pioneerml.data_loader.loaders.structured.graph.graph_loader import GraphLoader


class MultiLevelGraphLoader(GraphLoader):
    """Graph loader base that carries optional slice-level structure and targets."""

    def data_struct_fields(self) -> tuple[str, ...]:
        fields = list(super().data_struct_fields())
        fields.extend(
            [
                "node_slice_id",
                "slice_graph_id",
                "slice_ptr",
                "graph_slice_ptr",
                "atar_slice_ptr",
                "y_slice",
                "y_event",
                "atar_node_pdg_target",
                "atar_true_event_id",
                "atar_slice_pdg_target",
                "atar_slice_multi_target",
                "atar_slice_trigger_target",
                "atar_slice_start_target",
                "atar_slice_stop_target",
                "atar_angle_target",
                "atar_pion_stop_target",
                "atar_pion_stop_valid_target",
                "positron_initial_energy_target",
                "lyso_fracs_target",
                "lyso_payload_target",
                "lyso_mask_target",
                "is_trigger_target",
                "has_trigger_positron",
            ]
        )
        return tuple(dict.fromkeys(fields))

    def empty_data(self):
        data = super().empty_data()
        event_dim = max(0, int(self.empty_graph_target_dim()))
        data.node_slice_id = torch.empty((0,), dtype=torch.int64)
        data.slice_graph_id = torch.empty((0,), dtype=torch.int64)
        data.slice_ptr = torch.empty((0,), dtype=torch.int64)
        data.graph_slice_ptr = torch.empty((0,), dtype=torch.int64)
        data.atar_slice_ptr = torch.empty((0,), dtype=torch.int64)
        data.y_slice = torch.empty((0, event_dim), dtype=torch.float32)
        data.y_event = torch.empty((0, event_dim), dtype=torch.float32)
        data.atar_node_pdg_target = torch.empty((0, 3), dtype=torch.float32)
        data.atar_true_event_id = torch.empty((0,), dtype=torch.int64)
        data.atar_slice_pdg_target = torch.empty((0, 3), dtype=torch.float32)
        data.atar_slice_multi_target = torch.empty((0,), dtype=torch.float32)
        data.atar_slice_trigger_target = torch.empty((0,), dtype=torch.float32)
        data.atar_slice_start_target = torch.empty((0, 3), dtype=torch.float32)
        data.atar_slice_stop_target = torch.empty((0, 3), dtype=torch.float32)
        data.atar_angle_target = torch.empty((0, 3), dtype=torch.float32)
        data.atar_pion_stop_target = torch.empty((0, 3), dtype=torch.float32)
        data.atar_pion_stop_valid_target = torch.empty((0,), dtype=torch.float32)
        data.positron_initial_energy_target = torch.empty((0, 1), dtype=torch.float32)
        data.lyso_fracs_target = torch.empty((0, 20), dtype=torch.float32)
        data.lyso_payload_target = torch.empty((0, 20, 4), dtype=torch.float32)
        data.lyso_mask_target = torch.empty((0, 20), dtype=torch.float32)
        data.is_trigger_target = torch.empty((0,), dtype=torch.float32)
        data.has_trigger_positron = torch.empty((0,), dtype=torch.float32)
        data.x = data.x_node
        data.batch = torch.empty((0,), dtype=torch.int64)
        data.num_slices = 0
        return data

    def _slice_chunk_batch(self, chunk: dict, g0: int, g1: int):
        d = super()._slice_chunk_batch(chunk, g0, g1)

        node_ptr = chunk["node_ptr"]
        n0 = int(node_ptr[g0].item())
        n1 = int(node_ptr[g1].item())

        graph_slice_ptr = chunk.get("graph_slice_ptr")
        if graph_slice_ptr is not None:
            s0 = int(graph_slice_ptr[g0].item())
            s1 = int(graph_slice_ptr[g1].item())
        else:
            s0 = 0
            s1 = 0

        if "node_slice_id" in chunk and chunk["node_slice_id"] is not None:
            node_slice_id = chunk["node_slice_id"][n0:n1].to(dtype=torch.int64)
            if s1 > s0:
                node_slice_id = node_slice_id - int(s0)
            d.node_slice_id = node_slice_id
        else:
            d.node_slice_id = torch.empty((n1 - n0,), dtype=torch.int64)

        if "slice_graph_id" in chunk and chunk["slice_graph_id"] is not None and s1 > s0:
            d.slice_graph_id = (chunk["slice_graph_id"][s0:s1] - int(g0)).to(dtype=torch.int64)
        else:
            d.slice_graph_id = torch.empty((0,), dtype=torch.int64)

        if "slice_ptr" in chunk and chunk["slice_ptr"] is not None and s1 >= s0:
            d.slice_ptr = (chunk["slice_ptr"][s0 : s1 + 1] - int(n0)).to(dtype=torch.int64)
        else:
            d.slice_ptr = torch.empty((0,), dtype=torch.int64)

        if "graph_slice_ptr" in chunk and chunk["graph_slice_ptr"] is not None:
            d.graph_slice_ptr = (chunk["graph_slice_ptr"][g0 : g1 + 1] - int(s0)).to(dtype=torch.int64)
        else:
            d.graph_slice_ptr = torch.empty((0,), dtype=torch.int64)

        if "y_slice" in chunk and chunk["y_slice"] is not None and s1 > s0:
            d.y_slice = chunk["y_slice"][s0:s1]
        else:
            d.y_slice = torch.empty((0, self.empty_graph_target_dim()), dtype=torch.float32)

        if "y_event" in chunk and chunk["y_event"] is not None:
            d.y_event = chunk["y_event"][g0:g1]
        else:
            d.y_event = d.y_graph

        if "atar_node_pdg_target" in chunk and chunk["atar_node_pdg_target"] is not None:
            d.atar_node_pdg_target = chunk["atar_node_pdg_target"][n0:n1]
        else:
            d.atar_node_pdg_target = torch.empty((n1 - n0, 3), dtype=torch.float32)

        if "atar_true_event_id" in chunk and chunk["atar_true_event_id"] is not None:
            d.atar_true_event_id = chunk["atar_true_event_id"][n0:n1].to(dtype=torch.int64)
        else:
            d.atar_true_event_id = torch.empty((n1 - n0,), dtype=torch.int64)

        if "lyso_fracs_target" in chunk and chunk["lyso_fracs_target"] is not None:
            d.lyso_fracs_target = chunk["lyso_fracs_target"][n0:n1]
        else:
            d.lyso_fracs_target = torch.empty((n1 - n0, 20), dtype=torch.float32)

        if "is_trigger_target" in chunk and chunk["is_trigger_target"] is not None:
            d.is_trigger_target = chunk["is_trigger_target"][n0:n1]
        else:
            d.is_trigger_target = torch.empty((n1 - n0,), dtype=torch.float32)

        if "lyso_payload_target" in chunk and chunk["lyso_payload_target"] is not None:
            d.lyso_payload_target = chunk["lyso_payload_target"][g0:g1]
        else:
            d.lyso_payload_target = torch.empty((num_graphs, 20, 4), dtype=torch.float32)

        if "lyso_mask_target" in chunk and chunk["lyso_mask_target"] is not None:
            d.lyso_mask_target = chunk["lyso_mask_target"][g0:g1]
        else:
            d.lyso_mask_target = torch.empty((num_graphs, 20), dtype=torch.float32)

        if "positron_initial_energy_target" in chunk and chunk["positron_initial_energy_target"] is not None:
            d.positron_initial_energy_target = chunk["positron_initial_energy_target"][g0:g1]
        else:
            d.positron_initial_energy_target = torch.empty((num_graphs, 1), dtype=torch.float32)

        if "has_trigger_positron" in chunk and chunk["has_trigger_positron"] is not None:
            d.has_trigger_positron = chunk["has_trigger_positron"][g0:g1]
        else:
            d.has_trigger_positron = torch.zeros((num_graphs,), dtype=torch.float32)

        atar_slice_ptr = chunk.get("atar_slice_ptr")
        if atar_slice_ptr is not None:
            a0 = int(atar_slice_ptr[g0].item())
            a1 = int(atar_slice_ptr[g1].item())
            d.atar_slice_ptr = (atar_slice_ptr[g0 : g1 + 1] - int(a0)).to(dtype=torch.int64)
        else:
            a0 = 0
            a1 = 0
            d.atar_slice_ptr = torch.empty((0,), dtype=torch.int64)

        if "atar_slice_pdg_target" in chunk and chunk["atar_slice_pdg_target"] is not None and a1 > a0:
            d.atar_slice_pdg_target = chunk["atar_slice_pdg_target"][a0:a1]
        else:
            d.atar_slice_pdg_target = torch.empty((0, 3), dtype=torch.float32)
        if "atar_slice_multi_target" in chunk and chunk["atar_slice_multi_target"] is not None and a1 > a0:
            d.atar_slice_multi_target = chunk["atar_slice_multi_target"][a0:a1]
        else:
            d.atar_slice_multi_target = torch.empty((0,), dtype=torch.float32)
        if "atar_slice_trigger_target" in chunk and chunk["atar_slice_trigger_target"] is not None and a1 > a0:
            d.atar_slice_trigger_target = chunk["atar_slice_trigger_target"][a0:a1]
        else:
            d.atar_slice_trigger_target = torch.empty((0,), dtype=torch.float32)
        if "atar_slice_start_target" in chunk and chunk["atar_slice_start_target"] is not None and a1 > a0:
            d.atar_slice_start_target = chunk["atar_slice_start_target"][a0:a1]
        else:
            d.atar_slice_start_target = torch.empty((0, 3), dtype=torch.float32)
        if "atar_slice_stop_target" in chunk and chunk["atar_slice_stop_target"] is not None and a1 > a0:
            d.atar_slice_stop_target = chunk["atar_slice_stop_target"][a0:a1]
        else:
            d.atar_slice_stop_target = torch.empty((0, 3), dtype=torch.float32)
        if "atar_angle_target" in chunk and chunk["atar_angle_target"] is not None and a1 > a0:
            d.atar_angle_target = chunk["atar_angle_target"][a0:a1]
        else:
            d.atar_angle_target = torch.empty((0, 3), dtype=torch.float32)
        if "atar_pion_stop_target" in chunk and chunk["atar_pion_stop_target"] is not None and a1 > a0:
            d.atar_pion_stop_target = chunk["atar_pion_stop_target"][a0:a1]
        else:
            d.atar_pion_stop_target = torch.empty((0, 3), dtype=torch.float32)
        if "atar_pion_stop_valid_target" in chunk and chunk["atar_pion_stop_valid_target"] is not None and a1 > a0:
            d.atar_pion_stop_valid_target = chunk["atar_pion_stop_valid_target"][a0:a1]
        else:
            d.atar_pion_stop_valid_target = torch.empty((0,), dtype=torch.float32)

        d.x = d.x_node
        d.batch = d.node_graph_id
        d.edge_attr = d.x_edge
        d.num_slices = int(max(0, s1 - s0))
        return d
