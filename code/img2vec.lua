--[[
Written by: Andrej Karpathy & Justin Johnson (github.com/karpathy/neuraltalk2) 
Original name: eval.lua 
Modified by: Erick Cobos T (a01184587@itesm.mx)
Date: 13-July-2016

Passes images through a pretrained (and fine-tuned) VGG-net and saves the 
convnet activations (feature vectors) to file. 

Example:
		$ th img2vec.lua -image_folder /path/to/image/folder -model /path/to/model -num_images XX
	where XX is the total number of images to process. See code for defaults.

Note:
	To assert that rows in the feature vector matrix correspond to the images I 
	want, I modified misc/DataLoaderRaw.lua to read images in sequential order 
	(img_1.png, img_2.png, ...)

	Needs to be run in the neuraltalk2 folder.
--]]

require 'torch'
require 'hdf5'
-- local imports
require 'misc.DataLoaderRaw'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Save CNN feature vectors for every image in folder')
cmd:text()
cmd:text('Options')

-- Basic options
cmd:option('-model', 'models/model_id1-501-1448236541.t7', 'path to the pretrained model')
cmd:option('-num_images', 108000, 'how many images to process')
cmd:option('-image_folder', 'images/', 'folder where to find the images')
cmd:option('-feats_filename', 'full_feats.h5', 'name of the file where feats are saved')
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
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoaderRaw{folder_path = opt.image_folder, coco_json = ''}

-------------------------------------------------------------------------------
-- Load the network from  model checkpoint
-------------------------------------------------------------------------------
assert(string.len(opt.model) > 0, 'must provide a model')
local checkpoint = torch.load(opt.model)
local protos = checkpoint.protos
if opt.gpuid >= 0 then 
	for k,v in pairs(protos) do v:cuda() end 
end

-------------------------------------------------------------------------------
-- Run the CNN and store every feature vector
-------------------------------------------------------------------------------
protos.cnn:evaluate()
loader:resetIterator() -- rewind iterator back to first datapoint

local all_feats = torch.Tensor(opt.num_images, 768)

for n = 1, opt.num_images do
	-- fetch a batch of data and preprocess images
	local data = loader:getBatch{batch_size = 1}
	data.images = net_utils.prepro(data.images, false, opt.gpuid >= 0)

	-- forward the model to get loss
	local feats = protos.cnn:forward(data.images)
	
	-- save feats for the image
	all_feats[{n, {}}] = feats:double()

	-- print reports
	if n % 10 == 0 then
		print(string.format('%d/%d', n, opt.num_images))
	end
end


-- save hdf5 file with features
local feats_file = hdf5.open(opt.feats_filename, 'w')
feats_file:write('feats', all_feats)
feats_file:close()	
print('Done!')
