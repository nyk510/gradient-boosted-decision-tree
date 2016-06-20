import numpy as np
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self,determinator=None,children=None,value=None):
        """
        determinator: input x and return 0 or 1 function.
        children:child Nodes have 2 length.if Node has children, __call__ fnc return child followed by determinator function.
        value: this node's value. If this has no child node, __call__ fnc return value.
        """
        self.determinator = determinator
        self.children = children
        self.value = value
        self.is_leaf = True

    def __call__(self,x):

        # if leaf node
        if self.is_leaf:
            return self.value

        # else return child
        y = self.determinator(x)
        return self.children[y](x)

    def append_children(self,determinator,children):
        self.determinator = determinator
        self.children = children
        self.is_leaf = False
        return



class DecisionTree(object):

    def __init__(self,x,t,max_depth=8):
        self.x = x
        self.t = t
        self.nodes = [Node(value=.0),]
        self.leaf_index = [True,]
        self.max_depth = max_depth

    def __call__(self,x):
        y = []
        for x_i in x:
             y.append(self.nodes[0](x))
        return np.array(y)

    def glow(self):
        self.y = self(self.x)
        target = self.t - self.y



if __name__ == '__main__':
    np.random.seed = 71
    x = np.random.normal(size=10)
    t = np.sin(x) + np.random.normal(scale=.1,size=len(x))
    dtree = DecisionTree(x=x,t=t)
    print(dtree(x))
    plt.plot(x,t,'o')
    plt.show()
