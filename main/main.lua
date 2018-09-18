--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]
require 'nngraph'
require 'optim'
require 'debug'
torch.setdefaulttensortype('torch.FloatTensor')
seqmatchseq = {}
tr = require '../util/loadFiles'
include '../util/utils.lua'

include '../models/Embedding.lua'
include '../models/LSTM.lua'
include '../models/LSTMwwatten.lua'
include '../models/CNNwwSimatten.lua'
include '../models/pointNet.lua'

include '../nn/CAddRepTable.lua'
include '../nn/DMax.lua'

include '../snli/mLSTM.lua'
include '../snli/compAggSNLI.lua'

include '../squad/boundaryMPtr.lua'
include '../squad/sequenceMPtr.lua'

include '../wikiqa/compAggWikiqa.lua'

print ("require done !")

cmd = torch.CmdLine()
cmd:option('-batch_size',30,'number of sequences to train on in parallel')
cmd:option('-max_epochs',10,'number of full passes through the training data')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-reg',0,'regularize value')
cmd:option('-learning_rate',0.001,'learning rate')
cmd:option('-emb_lr',0 ,'embedding learning rate')
cmd:option('-emb_partial',true,'only update the non-pretrained embeddings')
cmd:option('-lr_decay',0.95,'learning rate decay ratio')
cmd:option('-dropoutP',0.3,'dropout ratio')
cmd:option('-expIdx', 0, 'experiment index')
cmd:option('-num_classes', 3, 'number of classes')


cmd:option('-wvecDim',313,'embedding dimension')
cmd:option('-mem_dim', 300, 'state dimension')
cmd:option('-att_dim', 300, 'attenion dimension')

cmd:option('-model','mLSTM','model')
cmd:option('-task','snli','task')

cmd:option('-comp_type', 'submul', 'w-by-w type')

cmd:option('-preEmb','glove','Embedding pretrained method')  -- Change Embedding model DO I'VE TO CHANGE THSI EVERY TIME?
cmd:option('-grad','adamax','gradient descent method')

cmd:option('-log', 'nothing', 'log message')


local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setnumthreads(1)

tr:init(opt)

local vocab = tr:loadVocab(opt.task)
local ivocab = tr:loadiVocab(opt.task)
opt.numWords = #ivocab
print ("Vocal size: "..opt.numWords)
print('loading data ..')
local train_dataset = tr:loadData('train', opt.task)
local test_dataset
if opt.task == 'snli' or opt.task == 'wikiqa' then test_dataset = tr:loadData('test', opt.task) end
local dev_dataset = tr:loadData('dev', opt.task)
torch.manualSeed(opt.seed)

local model_class = seqmatchseq[opt.model]

local model = model_class(opt)

local recordTrain, recordTest, recordDev
for i = 1, opt.max_epochs do
    model:train(train_dataset)
    model.optim_state['learningRate'] = model.optim_state['learningRate'] * opt.lr_decay

    recordDev = model:predict_dataset(dev_dataset)
    model:save('../trainedmodel/', opt, {recordDev}, i)
    if i == opt.max_epochs then
        model.params:copy( model.best_params )
        recordDev = model:predict_dataset(dev_dataset)
        if opt.task == 'snli' or opt.task == 'wikiqa' then recordTest   = model:predict_dataset(test_dataset) end
        recordTrain  = model:predict_dataset(train_dataset)
        model:save('../trainedmodel/', opt, {recordDev, recordTest, recordTrain}, i)
    end
end
