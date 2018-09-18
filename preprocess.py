# Carmen Dibut 
#MCs Information Technology
#Declaration  ....  
    
def prepSQuAD():
    reload(sys)
    sys.setdefaultencoding('utf-8')
    import json
    from nltk.tokenize import word_tokenize
    count = 0
    filenames = ['dev', 'train']
    for filename in filenames:
        fpr = open("data/squad/"+filename+"-v1.1.json", 'r')
        body = fpr.read()
        js = json.loads(body)
        #data = js["data"]
        #js = { "version" : js["data"], "data":data[0:subset] }
        #data = js["data"]
        fpw = open("data/squad/sequence/"+filename+".txt", 'w')
        for c in js["data"]:
            for p in c["paragraphs"]:
                context = p["context"].split(' ')
                context_char = list(p["context"])
                context_pos = {}
                for qa in p["qas"]:                       
                    question = word_tokenize(qa["question"])
                    question_tag = nltk.pos_tag(question)
                    
                    qtags = []
                    for words, postag in question_tag:
                        qtags.append(words +' '+ postag)

                    if filename == 'train':
                        for a in qa['answers']:
                            answer = a['text'].strip()
                            answer_start = int(a['answer_start'])

                        #add '.' here, just because NLTK is not good enough in some cases
                        answer_words = word_tokenize(answer+'.')
                        if answer_words[-1] == '.':
                            answer_words = answer_words[:-1]
                        else:
                            answer_words = word_tokenize(answer)

                        ptags = []
                        prev_context_words = word_tokenize( p["context"][0:answer_start ] )
                        prev_tag = nltk.pos_tag(prev_context_words)
                        for w, ppos in prev_tag:
                            ptags.append(w + ' ' + ppos)

                        ltags = []
                        left_context_words = word_tokenize( p["context"][answer_start:] )
                        left_tag = nltk.pos_tag(left_context_words)
                        for w, lpos in left_tag:
                            ltags.append(w + ' ' + lpos)
                        
                        
                        answer_reproduce = []
                        for i in range(len(answer_words)):
                            if i < len(left_context_words):
                                w = left_context_words[i]
                                answer_reproduce.append(w)
                        join_a = ' '.join(answer_words)
                        join_ar = ' '.join(answer_reproduce)

                        #if not ((join_ar in join_a) or (join_a in join_ar)):
                        if join_a != join_ar:
                            count += 1

                        fpw.write(' '.join(ptags + ltags) + '\t')
                        fpw.write(' '.join(qtags) + '\t')

                        pos_list = []
                        for i in range(len(answer_words)):
                            if i < len(left_context_words):
                                pos_list.append(str(len(prev_context_words)+i+1))
                        if len(pos_list) == 0:
                            print (join_ar)
                            print (join_a)
                            print ('answer:'+answer)
                        assert(len(pos_list) > 0)
                        fpw.write(' '.join(pos_list) + '\n')

                    else:
                        ctagDev = []
                        contextDev = word_tokenize(p["context"])
                        contextPOS = nltk.pos_tag(contextDev)
                        for w, pos in contextPOS:
                            ctagDev.append(w + ' ' + pos)
                            
                        fpw.write(' '.join(ctagDev) + '\t')
                        fpw.write(' '.join(qtags) + '\n')

        fpw.close()
        
    print ('SQuAD preprossing finished!')

prepSQuAD()
