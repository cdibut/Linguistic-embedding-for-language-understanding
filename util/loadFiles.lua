--[[
Copyright 2015 Singapore Management University (SMU). All Rights Reserved.

Permission to use, copy, modify and distribute this software and its documentation for purposes of research, teaching and general academic pursuits, without fee and without a signed licensing agreement, is hereby granted, provided that the above copyright statement, this paragraph and the following paragraph on disclaimer appear in all copies, modifications, and distributions.  Contact Singapore Management University, Intellectual Property Management Office at iie@smu.edu.sg, for commercial licensing opportunities.

This software is provided by the copyright holder and creator “as is” and any express or implied warranties, including, but not Limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed.  In no event shall SMU or the creator be liable for any direct, indirect, incidental, special, exemplary or consequential damages, however caused arising in any way out of the use of this software.
]]
--[[
Carmen Dibut
MSc IT

I modified the following functions:

function tr:buildVocab(task)
function tr:buildVacab2Emb(opt)
function tr:initialUNK(embRec, emb, opt)
function tr:buildData(filename, task)
]]
local stringx = require 'pl.stringx' -- module: pl.stringx 
require 'debug'
require 'paths'

tr = {}
tr.__index = tr
function tr:init(opt)
    if not paths.filep("../data/".. opt.task .."/vocab.t7") then
        self:buildVocab(opt.task)
    end

    if not paths.filep("../data/".. opt.task .."/sequence/train.t7") then
	if opt.task == 'snli' then
            tr:buildData('dev', opt.task)
            tr:buildData('test', opt.task)
            tr:buildData('train', opt.task)
        else
            tr:buildData('all', opt.task)
        end
    end

    if not paths.filep("../data/".. opt.task .."/initEmb.t7") then
        self:buildVacab2Emb(opt)
    end

end

function tr:loadVocab(task)
	print("loadVocab")
    return  torch.load("../data/".. task .."/vocab.t7")
end


function tr:loadUnUpdateVocab(task)
    return  torch.load("../data/".. task .."/unUpdateVocab.t7")
end

function tr:loadiVocab(task) -- index vocabulary used in mainDt to create opt.numWords
    print("LoadiVocab")
	return torch.load("../data/"..task.."/ivocab.t7")
end

function tr:loadVacab2Emb(task) -- create binary file initEmb.t7
    print("Loading embedding ...")
    return torch.load("../data/"..task.."/initEmb.t7")
end

function tr:loadData(filename, task)
    print("Loading data "..filename.."...") -- load binary file train.t7
    return torch.load("../data/"..task.."/sequence/"..filename..".t7")
end
-- Building Vocabularies to embedding 
function tr:buildVocab(task) -- build a dictionary based on words appear in glove If words saved then in vocab.t7, if not in glove then saved them in ivocab.t7
    print ("Building vocab dict ...") 
    if task == 'squad' then
	    local filenames = {dev="../data/"..task.."/sequence/dev.txt", train="../data/"..task.."/sequence/train.txt"}
        local vocab = {}
        local ivocab = {}
        local a_vocab = {}
        local a_ivocab = {}
        for _, filename in pairs(filenames) do -- for every line
	        for line in io.lines(filename) do
                local divs = stringx.split(line, '\t')
                    for j = 1, 2 do   -- words in left/ right  --- 
                    local words = stringx.split(divs[j], ' ')
                    for i = 1, #words, 2 do  
			            local word = words[i]
			            local pos = words[i+1]   -- if we enncounter the word for the fist time
                        local wp = word .. ' ' ..pos 
                        if vocab[wp] == nil then        
							local index = #ivocab + 1
                            vocab[wp] = index 
                            ivocab[index] = wp 
                        end
                    end
                end
            end
        end
		print(ivocab)
		torch.save("../data/"..task.."/vocab.t7", vocab) -- vocabulary: words to index dictionary
   		torch.save("../data/"..task.."/ivocab.t7", ivocab)  -- index to word
	end
end
-- Creating vocabulary to embedding
function tr:buildVacab2Emb(opt)
    local vocab = self:loadVocab(opt.task)
    local ivocab = self:loadiVocab(opt.task)
    local emb = torch.randn(#ivocab, opt.wvecDim) * 0.05 -- Initialising vocab Embedding
	if opt.task ~= 'snli' then emb:zero() end

    print ("Loading ".. opt.preEmb .. " ...") -- Loading word representation: Glove / word2vecGoogle / word2vecWikipedia
    local file 
    if opt.preEmb == 'glove' then
        file = io.open("../data/"..opt.preEmb.."/glove.840B.300d.txt", 'r')
		end
    if opt.preEmb == 'word2vecGoogle' then
        file = io.open("../data/"..opt.preEmb.."/GoogleNews-vectors-negative300.txt", 'r')
		end
    if opt.preEmb == 'word2vecWikipedia' then
        file = io.open("../data/"..opt.preEmb.."/wiki.en.text.vector", 'r')
    	end
    -- List of POS taggers
	local all_pos = {"NN","NNP","IN","DT","JJ","NNS","CC","VBD","CD","VBN","RB","VBZ","VB","TO","VBP","PRP","VBG","PRP$","POS","WDT","MD","WRP","NNPS","JJS","JJR","WP","EX","RBS","RBR","RP","WP$"}

    local count = 0
    local embRec = {}
    while true do
        local line = file:read()
        if line == nil then break end 
        local vals = stringx.split(line, ' ') 	
		local glove_word = vals[1] --  Embedding Vocab
		for ip = 1, #all_pos do
			local pos = all_pos[ip]
			local wp = glove_word .. ' ' .. pos -- POS Vocabulary 		
		    if vocab[wp] ~= nil then
				--print(vocab[wp])
				local index  = vocab[wp]
				for i = 2, #vals do

					emb[index] [i-1] = tonumber(vals[i]) 	--  copy embedding vector into emb
				end
			    -- requirements to Embedding
				emb[index][ #vals + ip -1 ] = 1				

				embRec[index] = 1 
				count = count + 1
				if count == #ivocab then
					break	
        		end
			end
    	end
	end
    print("Number of words Vocab ".. opt.preEmb .. ": "..(#vocab) )
    print("Number of words iVocab ".. opt.preEmb .. ": "..(#ivocab) )
    print("Number of words not appear in ".. opt.preEmb .. ": "..(#ivocab - count) )  

    torch.save("../data/"..opt.task.."/initEmb.t7", emb)
    torch.save("../data/"..opt.task.."/unUpdateVocab.t7", embRec)
end
-- Setting out of vocabulary words to random vector.
function tr:initialUNK(embRec, emb, opt)
    print("Initializing not appeared words ...")
    local windowSize = 4
    local numRec = {}
    local filenames = {'train', 'dev', 'test'}  -- train and dev have changed From Words to Word POS  
    local sentsnames = {'lsents', 'rsents'}
    for _, filename in pairs(filenames) do  
        local data = tr:loadData(filename, opt.task) --ivobab check files train and dev 
        for _, sentsname in pairs(sentsnames) do  -- left/right sentences			
            for _, sent in pairs(data[sentsname]) do
                for i = 1, sent:size(1) do
                    local word = sent[i]
                    if embRec[word] == nil then
                        if numRec[word] == nil then
                            numRec[word] = 0
                        end
                        local count = 0
                        for j = -windowSize, windowSize do
                            if i + j <= sent:size(1) and i + j >= 1 and embRec[sent[i+j]] ~= nil then
                                emb[word] = emb[word] + emb[sent[i+j]]
                                count = count + 1
                            end
                        end
                        numRec[word] = numRec[word] + count
                    end
                end
            end
        end
    end
    for k, v in pairs(numRec) do
        if v ~= 0 then
            emb[k] = emb[k] / (v+1)
        end
        print(v)
    end
end

function tr:buildData(filename, task)
    local trees = {}
    local lines = {}
    local dataset = {}
    idx = 1
    vocab = tr:loadVocab(task)
    print ("Building "..task.." "..filename.." data ...")
		
    if task == 'squad' then 
        local filenames = {dev="../data/"..task.."/sequence/dev.txt", train="../data/"..task.."/sequence/train.txt"}	
        for folder, filename in pairs(filenames) do
            local data = {}			
            for line in io.lines(filename) do
                local divs = stringx.split(line, '\t')
                local instance = {}
                for j = 1, 2 do	
                    local words = stringx.split(divs[j], ' ')
					instance[j] = torch.IntTensor(#words/2)  
				    for k =1, #words,2 do  
						local word = words[k]
						local pos = words[k+1]
						wp = word .. ' ' .. pos
			           	instance[j][(k + 1)/2] = vocab[ wp]

					end
			     end
                if folder == 'train' then
                -- For each answer position 
                    local ans_pos = stringx.split(stringx.strip(divs[3]),' ')
					if ans_pos ~= nil then 
                    	instance[3] = torch.IntTensor(#ans_pos+1)
                    	for i = 1, #ans_pos do
                        	instance[3][i] = tonumber(ans_pos[i])
                   		end
                    	instance[3][#ans_pos+1] = instance[1]:size(1)+1
					else
					    print("WTF")
					end
                end

                data[#data+1] = instance
            end
			print("saving torch data")
            torch.save("../data/"..task.."/sequence/"..folder..'.t7', data)
        end
    return dataset
	end
end

return tr
