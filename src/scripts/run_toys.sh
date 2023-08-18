python3 main.py --data_name Toys_and_Games --cf_weight 0.1 --gcl_weight 0.2 \
--model_idx exp_0 --gpu_id 0 --temperature 1 --graph_temp 0.2 \
--batch_size 256 --contrast_type Hybrid \
--num_intent_cluster 256 --seq_representation_type mean \
--num_hidden_layers 3 --pe --att_bias 1 --intent_cf_weight 0.1