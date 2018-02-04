import nltk
import string

with open('new_failing_forward.txt', 'r') as f:
    sample = f.read()
#print sample
f = open("new_failing_forward.txt", 'r')
#print f.read()

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

st = StanfordNERTagger('C:/Users/Piyush/Desktop/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz',
					   'C:/Users/Piyush/Desktop/stanford-ner-2014-06-16/stanford-ner.jar',
					   encoding='utf-8')


from nltk.tag import pos_tag
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
#tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
stop_words = set(stopwords.words('english'))

l = sample.split(".")
propernouns = []
from unidecode import unidecode
for sentence in f.readlines():
    #sentence = unidecode(sentence)
    #exclude = set(string.punctuation)
    #sentence = ''.join(ch for ch in sentence if ch not in exclude)
    sentence = unicode(sentence, errors='ignore')
    #print sentence
    text = sentence 
    tokenized_text = word_tokenize(text)
    classified_text = st.tag(tokenized_text)
    for i,j in classified_text:
        if j != u'O':
            print i, j
    #sentence = sentence.decode('utf-8')
    #word_tokens = tokenizer.tokenize(sentence)

'''
    fsentence = [w for w in word_tokens if not w in stop_words]

    tagged_sent = pos_tag(fsentence)
    # [('Michael', 'NNP'), ('Jackson', 'NNP'), ('likes', 'VBZ'), ('to', 'TO'), ('eat', 'VB'), ('at', 'IN'), ('McDonalds', 'NNP')]

    #proper = [word for word,pos in tagged_sent if pos == 'NNP']
    # ['Michael','Jackson', 'McDonalds']
    for word,pos in tagged_sent:
        if pos == 'NNP':
            word = word.lower()
            word = unidecode(word)
            if wn.morphy(word, wn.VERB) is not None:
                pass
            #print "Else"
            else:
            #if pos == 'NNP':
                print "***" + word + "***", sentence
                break
'''
    #propernouns.append(proper)
#print propernouns
'''
sentences = nltk.sent_tokenize(sample)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names

entity_names = []
for tree in chunked_sentences:
    # Print results per sentence
    # print extract_entity_names(tree)

    entity_names.extend(extract_entity_names(tree))

# Print all entity names
#print entity_names

# Print unique entity names
print set(entity_names)
'''
