#encoding=utf8
import json
from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp
from pprint import pprint

class StanfordNLP:
    def __init__(self,ip="127.0.0.1",port=8080):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=(ip, port)))

    def parse(self, text):
        return json.loads(self.server.parse(text))

# if __name__ == '__main__':
nlp = StanfordNLP()
result = nlp.parse("Hello world!  It is so beautiful.")


# result = nlp.parse("The atom is a basic unit of matter, it consists of a dense central nucleus surrounded by a cloud of negatively charged electrons.")

# pprint(result['coref'])
pprint(result)

from nltk.tree import Tree
# tree = Tree.parse(result['sentences'][0]['parsetree'])
# tree = Tree.fromstring(result['sentences'][2]['parsetree'])
# pprint(tree)
# tree.draw()

# ==============================================================================
# import json
# from jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp
# from pprint import pprint

# class StanfordNLP:
#     def __init__(self):
#         self.server = ServerProxy(JsonRpc20(),
#                                   TransportTcpIp(addr=("127.0.0.1", 8080)))

#     def parse(self, text):
#         return json.loads(self.server.parse(text))

# nlp = StanfordNLP()
# result = nlp.parse("Hello world!  It is so beautiful.")
# pprint(result)

# from nltk.tree import Tree
# tree = Tree.parse(result['sentences'][0]['parsetree'])
# pprint(tree)
