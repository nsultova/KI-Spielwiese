
"""
Exercise 01 as found in
https://github.com/karpathy/minbpe/blob/master/exercise.md
Most basic PoC of BPE for my own understanding
https://en.wikipedia.org/wiki/Byte_pair_encoding
"""

def count_bigrams(text):
    """
    >>> count_bigrams("aaabdaaabac")
    {'aa': 4, 'ab': 2, 'bd': 1, 'da': 1, 'ba': 1, 'ac': 1}
    """
    bigram_occurence = {}
    for i in range(len(text) -1):
        bigram = text[i:i+2]
        if not bigram in bigram_occurence:
            bigram_occurence[bigram] = 0
        bigram_occurence[bigram] += 1
    return bigram_occurence

def most_frequent_bigram(text):
    """
    >>> most_frequent_bigram("aaabdaaabac")
    'aa'
    """
    max_count = 0
    max_bigram = None
    for bigram, count in count_bigrams(text).items():
        #print(bigram, count)
        if count > max_count:
            max_count = count
            max_bigram = bigram
        
    return max_bigram

class DummyTokenizer():
    def __init__(self, vocab_size=256, num_merges=3):
        self.vocab_size = vocab_size
        self.merges = num_merges
        self.vocab_map = dict([(i,chr(i)) for i in range(vocab_size)])

    def _find_next_vocab_idx(self):
        """
        Find the next free spot in vocab_map for inserting a non_terminal symbol mapping 
        to the bigram in question

        >>> DummyTokenizer()._find_next_vocab_idx()
        256

        >>> DummyTokenizer(vocab_size=512)._find_next_vocab_idx()
        512
        """
        return len(self.vocab_map)
    
    def _non_terminal_symbol(self, vocab_map_idx):
        """
        In our dummy case non-terminal symbols are [Z-A], each denoting a bigram to be substituted
        >>> DummyTokenizer()._non_terminal_symbol(257)
        'Y'
        """
        symbol_idx = vocab_map_idx - self.vocab_size
        ASCII_Z = 90
        return chr(ASCII_Z - symbol_idx) 

    def _merge(self, text):
        """
        >>> DummyTokenizer()._merge("aaabdaaabac")
        'ZabdZabac'
        """
        bigram = most_frequent_bigram(text)
        next_substitution = self._non_terminal_symbol(self._find_next_vocab_idx())
        text = text.replace(bigram, next_substitution)
        self.vocab_map[self._find_next_vocab_idx()] = bigram
        return text

    def train(self, text):
        """
        >>> DummyTokenizer().train("aaabdaaabac")
        'XdXac'
        """
        for i in range(self.merges):
            text = self._merge(text)
        return text 

    def encode(self, text):
        """
        >>> t = DummyTokenizer()
        >>> t.train("aaabdaaabac")
        'XdXac'
        >>> t.encode("aaabdaaabac")
        'XdXac'
        >>> t.encode("abcuuaab")
        'abcuuZb'
        """
        for (idx, bigram ) in list(self.vocab_map.items())[256:]:
        #print(idx,bigram, non_terminal_symbol(idx))
            text = text.replace(bigram, self._non_terminal_symbol(idx))
        return text
    
    def decode(self, text):
        """ 
        >>> t = DummyTokenizer()
        >>> t.train("aaabdaaabac")
        'XdXac'
        >>> t.decode('XdXac')
        'aaabdaaabac'
        """
        reversed_mapping = list(self.vocab_map.items())[256:]; reversed_mapping.reverse()
        for (idx, bigram ) in reversed_mapping:
            text = text.replace(self._non_terminal_symbol(idx), bigram)
        #print(idx,bigram, non_terminal_symbol(idx))
        return text

if __name__ == "__main__":
    import doctest
    test_result = doctest.testmod()
    print(test_result)
