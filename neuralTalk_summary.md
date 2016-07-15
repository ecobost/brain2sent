# Checkpoint data
Contents of the pretrained checkpoint (model_id1-501-1448236541.t7):

* Training loss  
`loss_history = {#iteration: loss}`

* Predicted sentences on validation examples  
`val_predictions = {#iteration: {image_id, caption}}`

* Validation loss  
`val_loss_history = {#iteration: validation loss}`

* Language metrics  
`val_lang_stats_history = {#iteration: {Bleu_1, ROUGE_L, METEOR, Bleu_4, Bleu_3, Bleu_2, CIDEr}}`

* Vocabulary  
`vocab = {word_id: word}`

* Number of iterations  
`iter = 132500`

* Configuration options (see below)  
`opt = {option_name: value}`

* Actual trained models (see below)  
`protos = {cnn: Trained convnet, lm: Trained LSTM}`

Note: Use `checkpoint.xxxxx` to access contents

# LSTM architecture
Single layer, 768 input, hidden and cell vector, 9567 words in vocabulary.

# Convnet architecture
VGG-16 architecture with top layer replaced to produce a 768-dimensional output vector.  
Output of `print(checkpoint.protos.cnn)`:

```lua
nn.Sequential {
	[input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> (15) -> (16) -> (17) -> (18) -> (19) -> (20) -> (21) -> (22) -> (23) -> (24) -> (25) -> (26) -> (27) -> (28) -> (29) -> (30) -> (31) -> (32) -> (33) -> (34) -> (35) -> (36) -> (37) -> (38) -> (39) -> (40) -> output]
	(1): cudnn.SpatialConvolution(3 -> 64, 3x3, 1,1, 1,1)
	(2): cudnn.ReLU
	(3): cudnn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
	(4): cudnn.ReLU
	(5): cudnn.SpatialMaxPooling(2x2, 2,2)
	(6): cudnn.SpatialConvolution(64 -> 128, 3x3, 1,1, 1,1)
	(7): cudnn.ReLU
	(8): cudnn.SpatialConvolution(128 -> 128, 3x3, 1,1, 1,1)
	(9): cudnn.ReLU
	(10): cudnn.SpatialMaxPooling(2x2, 2,2)
	(11): cudnn.SpatialConvolution(128 -> 256, 3x3, 1,1, 1,1)
	(12): cudnn.ReLU
	(13): cudnn.SpatialConvolution(256 -> 256, 3x3, 1,1, 1,1)
	(14): cudnn.ReLU
	(15): cudnn.SpatialConvolution(256 -> 256, 3x3, 1,1, 1,1)
	(16): cudnn.ReLU
	(17): cudnn.SpatialMaxPooling(2x2, 2,2)
	(18): cudnn.SpatialConvolution(256 -> 512, 3x3, 1,1, 1,1)
	(19): cudnn.ReLU
	(20): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
	(21): cudnn.ReLU
	(22): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
	(23): cudnn.ReLU
	(24): cudnn.SpatialMaxPooling(2x2, 2,2)
	(25): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
	(26): cudnn.ReLU
	(27): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
	(28): cudnn.ReLU
	(29): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
	(30): cudnn.ReLU
	(31): cudnn.SpatialMaxPooling(2x2, 2,2)
	(32): nn.View(-1)
	(33): nn.Linear(25088 -> 4096)
	(34): cudnn.ReLU
	(35): nn.Dropout(0.500000)
	(36): nn.Linear(4096 -> 4096)
	(37): cudnn.ReLU
	(38): nn.Dropout(0.500000)
	(39): nn.Linear(4096 -> 768)
	(40): cudnn.ReLU
}
```

# Options
Output of `print(checkpoint.opt)`:

```lua
{
	cnn_optim_beta : 0.999
	finetune_cnn_after : 0
	batch_size : 16
	val_images_use : 3200
	optim_epsilon : 1e-08
	input_encoding_size : 768
	losses_log_every : 25
	id : "1-501-1448236541"
	optim_beta : 0.999
	input_h5 : "/scr/r6/karpathy/cocotalk.h5"
	rnn_size : 768
	cnn_learning_rate : 1e-05
	cnn_optim_alpha : 0.8
	language_eval : 1
	learning_rate_decay_every : 50000
	optim : "adam"
	gpuid : 0
	cnn_model : "model/VGG_ILSVRC_16_layers.caffemodel"
	drop_prob_lm : 0.75
	grad_clip : 0.1
	cnn_weight_decay : 0
	input_json : "/scr/r6/karpathy/cocotalk.json"
	seed : 123
	learning_rate_decay_start : -1
	seq_per_img : 5
	cnn_optim : "adam"
	max_iters : -1
	checkpoint_path : "checkpoints"
	start_from : "/scr/r6/karpathy/neuraltalk2_checkpoints/vgood1/model_id3-230-1448140513.t7"
	learning_rate : 0.0004
	cnn_proto : "model/VGG_ILSVRC_16_layers_deploy.prototxt"
	backend : "cudnn"
	save_checkpoint_every : 2500
	optim_alpha : 0.8
}
```
