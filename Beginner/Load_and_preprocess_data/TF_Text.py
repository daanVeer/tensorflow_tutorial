import tensorflow as tf
import tensorflow_text as text

docs = tf.constant([u'Evertything not saved will be lost.'.encode('UTF-16-BE'), u'Sad'.encode('UTF-16-BE')])
utf8_docs = tf.strings.unicode_transcode(docs, input_encoding='UTF-16-BE', output_encoding='UTF-8')

tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.', u'Sad'.encode('UTF-8')])
print(tokens.to_list())

# unicode script tokenizer

tokenizer = text.UnicodeScriptTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.', u'Sad'.encode('UTF-8')])
print(tokens.to_list())

# unicode split

tokens = tf.strings.unicode_split([u"仅今年前".encode('UTF-8')], 'UTF-8')
print(tokens.to_list())


tokenizer = text.UnicodeScriptTokenizer()
(tokens, offset_starts, offset_limits) = tokenizer.tokenize_with_offsets(['everything not saved will be lost.', u'Sad'.encode('UTF-8')])
print(tokens.to_list())
print(offset_starts.to_list())
print(offset_limits.to_list())

# example

docs = tf.data.Dataset.from_tensor_slices([['Never tell me the odds.'], ["it's a trap!"]])
tokenizer = text.WhitespaceTokenizer()
tokenized_docs = docs.map(lambda x: tokenizer.tokenize(x))
iterator = iter(tokenized_docs)
print(next(iterator).to_list())
print(next(iterator).to_list())

## word shape

tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['Everything not saved will be lost.', u'Sad'.encode('UTF-8')])

f1 = text.wordshape(tokens, text.WordShape.HAS_TITLE_CASE)
f2 = text.wordshape(tokens, text.WordShape.IS_UPPERCASE)
f3 = text.wordshape(tokens, text.WordShape.HAS_SOME_PUNCT_OR_SYMBOL)
f4 = text.wordshape(tokens, text.WordShape.IS_NUMERIC_VALUE)

print(f1.to_list())
print(f2.to_list())
print(f3.to_list())
print(f4.to_list())

## n-grams and windows

tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['Everything not saved will be lost.', u'Sad'.encode('UTF-8')])

bigrams = text.ngrams(tokens, 2, reduction_type = text.Reduction.STRING_JOIN)

print(bigrams.to_list())

