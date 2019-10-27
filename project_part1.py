import spacy
import pickle
from collections import Counter
import math
from itertools import chain,combinations


class InvertedIndex:
    def __init__(self):
        ## You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens = {}
        self.tf_entities = {}

        self.tf_norm_tokens = {}
        self.tf_norm_entities = {}

        ## You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens = {}
        self.idf_entities = {}


    def statistic(self,spacy_doc):
        lt = [token.text for token in spacy_doc]
        return dict(Counter(lt))

    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        # tokenize docs
        nlp = spacy.load("en_core_web_sm")
        for doc_id in documents:
            doc = nlp(documents[doc_id])
            tmp_tk = self.statistic(doc)
            tmp_en = self.statistic(doc.ents)
            #get all entities
            #ent is a class, need to store its text only in the dict, otherwise it's easy to produce error
            for ent in tmp_en:
                if ent in self.tf_entities:
                    if doc_id not in self.tf_entities[ent]:
                        self.tf_entities[ent][doc_id]=tmp_en[ent]
                else:
                    self.tf_entities[ent] = {doc_id:tmp_en[ent]}
            #get all tokens
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    if token.text in self.tf_entities and doc_id in self.tf_entities[token.text]:
                        pass
                    else:
                        if token.text in self.tf_tokens:
                            #doc_id exists, it means duplicate tokens in the same doc
                            #no need to add the frequence again
                            if doc_id not in self.tf_tokens[token.text]:
                                self.tf_tokens[token.text][doc_id]=tmp_tk[token.text]
                        else:
                            self.tf_tokens[token.text] = {doc_id:tmp_tk[token.text]}
                            # self.tf_norm_tokens[token.text] = {doc_id:1+math.log(1+math.log(tmp_tk[token]))}
        
        #construct dictionary of tf_norm
        #include doc not containing the term, assign 0 to count
        for token in self.tf_tokens:
            for doc_id in documents:
                if doc_id in  self.tf_tokens[token]:
                    if token in self.tf_norm_tokens:
                        self.tf_norm_tokens[token][doc_id] = 1+math.log(1+math.log(self.tf_tokens[token][doc_id]))
                    else:
                        self.tf_norm_tokens[token] = {doc_id:1+math.log(1+math.log(self.tf_tokens[token][doc_id]))}
                else:
                    if token in self.tf_norm_tokens:
                        self.tf_norm_tokens[token][doc_id] = 0
                    else:
                        self.tf_norm_tokens[token] = {doc_id:0}

        for ent in self.tf_entities:
            for doc_id in documents:
                #doc contains ent
                if doc_id in self.tf_entities[ent]:
                    if ent in self.tf_norm_entities:
                        self.tf_norm_entities[ent][doc_id] = 1+math.log(self.tf_entities[ent][doc_id])
                    else:
                        self.tf_norm_entities[ent] = {doc_id:1+math.log(self.tf_entities[ent][doc_id])}
                else:#doc does not contain ent
                    if ent in self.tf_norm_entities:
                        self.tf_norm_entities[ent][doc_id] = 0
                    else:
                        self.tf_norm_entities[ent] = {doc_id:0}
        
        #construct dictionary of idf
        for token in self.tf_tokens:
            self.idf_tokens[token] = 1+math.log(len(documents)/(1+len(self.tf_tokens[token])))

        for ent in self.tf_entities:
            self.idf_entities[ent] = 1+math.log(len(documents)/(1+len(self.tf_entities[ent])))
    
    # tell if a list is a sublist of another
    def isSubList(self,query_token_list,entity_token_lsit):
        count1 = dict(Counter(query_token_list))
        count2 = dict(Counter(entity_token_lsit))
        for x in count2:
            if x not in count1:
                return 0
            else:
                if count1[x] != count2[x]:
                    return 0
        return 1

    # get the complement of a list
    def getListComplement(self,query_token_list,entity_token_lsit):
        lc = []
        qtl_sorted = sorted(query_token_list)
        etl_sorted = sorted(entity_token_lsit)
        while etl_sorted:
            if qtl_sorted[0] < etl_sorted[0]:
                lc.append(qtl_sorted[0])
                qtl_sorted.pop(0)
            else:
                qtl_sorted.pop(0)
                etl_sorted.pop(0)
        return lc + qtl_sorted
            
    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):
        query_splits = []
        query_token = [qt for qt in Q.split(" ")]
        #get all subset of doe
        subset_pool = chain.from_iterable(combinations(list(DoE.keys()), r) for r in range(len(DoE)+1))
        for sb in subset_pool:
            if len(sb) == 0:
                query_splits.append({"tokens":query_token,"entities":[]})
                continue
            tmp = [y for x in sb for y in x.split(" ")]
            if(self.isSubList(query_token,tmp)):
                query_splits.append({"tokens":self.getListComplement(query_token,tmp),"entities":list(sb)})
        return query_splits

    def max_score_query(self, query_splits, doc_id):
        res = tuple()
        max_score = 0
        for query in query_splits:
            token_score = 0
            entities_score = 0
            for tk in query["tokens"]:
                #token not in the doc
                if tk not in self.tf_norm_tokens:
                    token_score += 0
                else:
                    token_score += self.tf_norm_tokens[tk][doc_id] * self.idf_tokens[tk]
            if len(query["entities"]) > 0:
                for et in query["entities"]:
                    if et not in self.tf_norm_entities:
                        entities_score += 0
                    else:
                        entities_score += self.tf_norm_entities[et][doc_id]*self.idf_entities[et]
            combined_score = token_score*0.4+entities_score
            if combined_score > max_score:
                max_score = combined_score
                res = (combined_score,query)
        return res



# documents = {1: 'According to Los Angeles Times, The Boston Globe will be experiencing another recession in 2020. However, The Boston Globe decales it a hoax.',
#              2: 'The Washington Post declines the shares of George Washington.',
#              3: 'According to Los Angeles Times, the UNSW COMP6714 students should be able to finish project part-1 now.'}

# index = InvertedIndex()
# index.index_documents(documents)

