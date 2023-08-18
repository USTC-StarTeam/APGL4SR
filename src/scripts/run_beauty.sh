python3 main.py --data_name Beauty --cf_weight 0.1 --gcl_weight 0.1 \
--model_idx exp_0 --gpu_id 0 --temperature 1 --graph_temp 0.2 \
--batch_size 256 --contrast_type Hybrid \
--num_intent_cluster 256 --seq_representation_type mean \
--warm_up_epoches 0 --num_hidden_layers 1 --pe --att_bias 1