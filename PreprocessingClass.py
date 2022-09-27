import spacy
"""
Blueprint of preprocessing and lemmatization for the Natural language 
processing

Args:
  doc: List of the sentences

Returns: 
  final_token : The list of the words

"""

class Preprocessing():
  def __init__(self, doc):
    """Inits the Preprocessing"""
    self.doc = doc
  
  def cleanData(self, doc):
    """Clean the data by removing stop words punctuvation"""

    nlp = spacy.load('en_core_web_sm')
    doc = doc.lower()
    doc = nlp(doc)
    tokens = [tokens.lower_ for tokens in doc]
    tokens = [tokens for tokens in doc if (tokens.is_stop == False)]
    tokens = [tokens for tokens in tokens if (tokens.is_punct == False)]
    # print(tokens)
    return tokens

  def lemmatization(self, tokens):
    """get the root words to reduce inflection of words"""
      # self.tokens = tokens
      final_token = [token.lemma_ for token in tokens]
      # print(" ".join(final_token))
      return final_token #" ".join(final_token)
 
  def run_all(self):
    """Run all the methods as per the requirments"""
    tokens = self.cleanData(self.doc)
    final_token = self.lemmatization(tokens)
    return final_token

doc = ("Find Alternatives To Pagerduty at Shopwebly, the Website to Compare Prices! Find and Compare Alternatives To Pagerduty Online. Save Now at Shopwebly! Many Products. Quick Results. Easy Access. Compare Products. Search and Discover   ")
preprocessing = Preprocessing(doc)
preprocessing.run_all()