#######  Doc2Vec  #########

import gensim
import gensim.downloader as api
"""
Doc2vec is an NLP tool for representing documents as a vector and is a generalizing of the word2vec method.
Args:
  doc: List of the sentences

Returns: 
  model : Doc2Vec Model

"""
class Doc2Vec():
  def __init__(self, doc):
    """Inits the Preprocessing"""
    self.doc = doc
  
  ######
  def tagged_document(self, list_of_list_of_words):
    """tagged the documents"""
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
    # data_for_training = list(tagged_document(doc))

    # return data_for_training

  def doc2vec_model_train(self,data_for_training):
    """doc2vec model"""
    # Initialise the Model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=40, min_count=2, epochs=30)

    # build the vocabulary
    model.build_vocab(data_for_training)

    # train the Doc2Vec model
    model.train(data_for_training, total_examples=model.corpus_count, epochs=model.epochs)

    return model
 
  def run_all(self):
    """Run all the methods as per the requirements"""
    # data_for_training = self.tagged_document(self.doc)
    model = self.doc2vec_model_train(list(self.tagged_document(self.doc)))
    return model


#Download the Dataset
dataset = api.load("text8")
data = [d for d in dataset]


doc2vec = Doc2Vec(data)
model = doc2vec.run_all()
model.infer_vector(['violent', 'means', 'to', 'destroy', 'the','organization'])
