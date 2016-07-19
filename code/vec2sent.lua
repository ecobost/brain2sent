--[[
Written by: Andrej Karpathy & Justin Johnson (github.com/karpathy/neuraltalk2) 
Original name: eval.lua 
Modified by: Erick Cobos T (a01184587@itesm.mx)
Date: 13-July-2016

Runs the RNN language model to generate phrases conditioned on 768-d vectors
(one per matrix row) and writes them to file: one sentence per line.

Example:
		$ th vec2sent.lua -model /path/to/model -feats /path/to/feature/vectors
	See code for defaults.

Note:
	Needs to be run in the neuraltalk2 folder.
--]]

require 'torch'
require 'hdf5'
-- local imports
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Generate sentences from CNN feature vectors')
cmd:text()
cmd:text('Options')

-- Input paths
cmd:option('-model', 'models/model_id1-501-1448236541.t7', 'path to the pretrained model')
cmd:option('-feats', 'feats.h5', 'file containing the feature vectors')
cmd:option('-sents_filename', 'sents.txt', 'filename to store sentences' )
-- Sampling options
cmd:option('-sample_max', 1, '1 = sample argmax words. 0 = sample from distributions.')
cmd:option('-beam_size', 2, 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
cmd:option('-temperature', 1.0, 'temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Load the matrix of feature vectors
-------------------------------------------------------------------------------
local feats_file = hdf5.open(opt.feats, 'r')
local feats_data = feats_file:read('feats'):all()
feats_file:close()

-------------------------------------------------------------------------------
-- Load the network from  model checkpoint
-------------------------------------------------------------------------------
assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
local vocab = checkpoint.vocab
local protos = checkpoint.protos
protos.lm:createClones() -- reconstruct clones inside the language model
if opt.gpuid >= 0 then 
	for k,v in pairs(protos) do v:cuda() end 
end


-------------------------------------------------------------------------------
-- Run the RNN and save sentences to file
-------------------------------------------------------------------------------
protos.lm:evaluate()
local sents_file = io.open(opt.sents_filename, "w")
local num_vectors = feats_data:size()[1]

for n = 1,num_vectors do
	-- get nth feature vector as batch
	local feats = feats_data[{{n},{}}]:cuda()

	-- forward the model to also get generated samples for each image
	local sample_opts = {sample_max = opt.sample_max, beam_size = opt.beam_size, temperature = opt.temperature}
	local seq = protos.lm:sample(feats, sample_opts)
	local sents = net_utils.decode_sequence(vocab, seq)

	-- write sentence to file
	sents_file:write(sents[1] .. '\n')

	-- print reports
	if n % 10 == 0 then
		print(string.format('%d/%d', n, num_vectors))
	end
end

-- close file
sents_file:close()
print('Done!')
