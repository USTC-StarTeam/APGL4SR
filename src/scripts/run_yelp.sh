python3 main.py --data_name Yelp --cf_weight 0.1 --gcl_weight 0.1 \
--model_idx exp_0 --gpu_id 0 --temperature 1 --graph_temp 0.2 \
--batch_size 256 --contrast_type Hybrid \
--seq_representation_type mean \
--num_hidden_layers 2 --att_bias 1 --pe \
--tao 0.4 --gamma 0.7 --beta 0.2