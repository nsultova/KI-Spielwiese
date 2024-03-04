class DummyTokenizer():
    def __init__(self):
        pass

    def train(self,text, vocab_size):
        pass

    def encode(self, text):
        return [258, 100, 258, 97, 99]
    
    def decode(self, text):
        return "aaabdaaabac"




def test():
    tokenizer = DummyTokenizer()
    text = "aaabdaaabac"
    tokenizer.train(text, 256 + 3) # 256 are the byte tokens, then do 3 merges
    print(tokenizer.encode(text))
    # [258, 100, 258, 97, 99]
    print(tokenizer.decode([258, 100, 258, 97, 99]))
    # aaabdaaabac

if __name__ == "__main__":
    test()
