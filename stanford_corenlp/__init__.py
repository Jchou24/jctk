import json
from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp
from pprint import pprint
from corenlp import StanfordCoreNLP

class StanfordNLP:
    def __init__(self,ip="127.0.0.1",port=8080):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=(ip, port)))

    def parse(self, text):
        return json.loads(self.server.parse(text))

class StanfordParser():
    def __init__(self, ip="127.0.0.1",port=8080):
        try:
            self.parser = StanfordNLP(ip,port)
        except:
            pass

    def start_remote_parser(self,ip="127.0.0.1",port=8080):
        self.parser = StanfordNLP(ip,port)

    def start_local_parser(self):
        self.parser = StanfordCoreNLP()  # wait a few minutes...

    def parse(self,text):
        try:
            return self.parser.parse(text)
        except:
            self.start_local_parser()
            self.parser.parse(text)

if __name__ == '__main__':
    parser = StanfordParser()
    pprint( parser.parse("Hello world!  It is so beautiful.") )
