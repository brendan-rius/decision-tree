import math
from collections import Counter, deque
import pygraphviz


class Node:
    def __init__(self, x, y, depth=0):
        """
        Create a node containing a dataset made of feature vectors (x) and their corresponding labels (y)
        :param x: the feature vectors of the training examples
        :param y: the labels of the training examples
        :param depth: the depth of the node
        """
        self.x = x
        self.y = y
        self.children = []  # The children of the node
        self.feature = None  # The feature on which the node splits into children
        self.depth = depth
        self.parent = None

    def entropy(self):
        """
        Compute the entropy log2 of the dataset of the node, based on their labels.
        This can vary between 0 (all labels are the same) to 1 (labels are different and equally distributed)
        :return: the entropy
        """

        def entropy_of_vector(vector):
            """
            Compute the entropy of a probability vector
            :param vector: the probability vector (example: [.1, .5, .4])
            :return: the entropy of the vector (log base 2)
            """
            entropy = 0
            for p in vector:
                if p != 0:  # log2(0) is not defined, but counts as 0 in the entropy
                    entropy += p * math.log2(p)
            return -entropy

        def construct_probability_vector():
            """
            Compute the frequency of each label in the dataset and put it in a vector.
            :return:
            """
            # We count the labels
            counter = Counter()
            for label in self.y:
                counter.update(label)

            # We divide the count of each label by the total number of data points in the set
            len_dataset = float(len(self.x))
            probability_vector = [count / len_dataset for count in counter.values()]

            return probability_vector

        return entropy_of_vector(construct_probability_vector())

    def generate_children_for_feature(self, feature):
        """
        Create children for the node based on a certain feature to split on.
        This does not add children to the nodes, it only returns them.
        This will generate one child for each outcome of the feature.
        For example, if you have a feature "color" that can contain 1 for blue, 2 for red, 3 for black (feature
        outcomes), thi will create three children node containing the corresponding correct dataset (one with only the
        blue, the other one with only the red, and the other one with only the black)
        :param feature: the feature to split on.
        :return: a dictionary with keys being the feature's outcomes and the values being the associated node.
        """
        children = {}
        for outcome in feature.outcomes:  # For each possible outcome of the feature
            outcome_x = []
            outcome_y = []
            # For each training example of the node
            for training_example_features, training_example_label in zip(self.x, self.y):
                # If the the example has the same outcome, we add it to the child node
                if outcome == feature.extract(training_example_features):
                    outcome_x.append(training_example_features)
                    outcome_y.append(training_example_label)
            outcome_node = Node(outcome_x, outcome_y)
            children[outcome] = outcome_node
        return children

    def choose_best_feature(self, features):
        """
        Choose the best feature to split the current node on in a set of features.
        We choose the feature that maximizes the information gain.
        :param features: the set of feature to choose from
        :return: the best feature (based on the information gain) to split on
        """

        if len(features) == 0:
            raise RuntimeError("There is no feature to choose from")

        best_feature = (None, None)  # (feature, information_gain)
        for feature in features:
            children = self.generate_children_for_feature(feature)
            gain = self.information_gain(children.values())
            if best_feature[1] is None or gain > best_feature[1]:
                best_feature = (feature, gain)
        return best_feature[0]

    def information_gain(self, children):
        """
        Compute the information gain that would occur we use a set of nodes as children nodes.
        :param children: the children nodes
        """
        entropy = self.entropy()
        children_weighted_entropy = sum([(len(child.x) / len(self.x)) * child.entropy() for child in children])
        return entropy - children_weighted_entropy

    def assign_children(self, children):
        """
        Assign children to the node and automatically set their depth
        :param children: the list of children
        """
        self.children = children
        for child in self.children.values():
            child.depth = self.depth + 1
            child.parent = self

    @property
    def label(self):
        """
        Return the most common label in the node's dataset.
        """
        if not self.y:
            return self.parent.label
        else:
            counter = Counter(self.y)
            return counter.most_common(1)[0][0]

    @property
    def splittable(self):
        """
        Check whether the node is splittable or not.
        The node is splittable only if it contains more than one element and if it contains different labels.
        """
        return 1 <= len(self.y) != self.y.count(self.y[0])

    @property
    def is_leaf(self):
        """
        Check whether the node has children or not
        """
        return len(self.children) == 0

    def __str__(self):
        datapoints = len(self.y)
        number_different_labels = len(set(self.y))
        if self.is_leaf:
            return "Leaf\n{} data points\n{} different labels".format(datapoints, number_different_labels)
        else:
            return "Split on feature #{}\n{} data points\n{} different labels".format(self.feature.feature_index,
                                                                                      datapoints,
                                                                                      number_different_labels)


class Feature:
    def __init__(self, feature_index, outcomes):
        """
        Create a feature based on the n-th component of a features-vector
        :param feature_index: n
        """
        self.feature_index = feature_index
        self.outcomes = outcomes

    @classmethod
    def build_features_from_dataset(cls, x):
        """
        Build a list of features from a dataset
        :param x: the features vector of the dataset
        :return: a list of features from this dataset
        """
        features = set()
        features_indices = range(len(x[0]))
        for feature_index in features_indices:
            outcomes = set()
            for training_example in x:
                outcomes.add(training_example[feature_index])
            features.add(Feature(feature_index, outcomes))
        return features

    def extract(self, vector):
        """
        Extract this feature value in a features vector
        :param vector: the features vector
        """
        return vector[self.feature_index]


class DecisionTreeClassifier:
    def __init__(self, max_depth=10):
        """
        Create a decision tree classifier
        :param max_depth: the maximum depth of the tree
        """
        self.tree = None
        self.max_depth = max_depth

    def fit(self, x, y):
        """
        Builds the tree from training data
        :param x: the features vectors of the data
        :param y: the corresponding labels
        """

        # We build all of the possible features for the training examples
        available_features = Feature.build_features_from_dataset(x)

        # We create the root node containing all the data
        root = Node(x, y)

        # We create the queue, because we will construct the tree breadth-first.
        # This queue contains the nodes that need to be expanded (the ones we need to find children for).
        # Each element of the queue if a tuple (node, available_features)
        queue = deque([(root, available_features)])
        while len(queue) > 0:
            node, current_depth = queue.pop()
            if node.splittable and node.depth < self.max_depth:
                best_feature = node.choose_best_feature(available_features)
                node.assign_children(node.generate_children_for_feature(best_feature))
                node.feature = best_feature
                for child in node.children.values():
                    # The child cannot use one of its parents' feature
                    available_features_for_child = available_features - {best_feature}
                    queue.append((child, available_features_for_child))

        self.tree = root

    def to_dot(self):
        graph = pygraphviz.AGraph(directed=True, strict=True)

        def iterate(node):
            if not node.is_leaf:
                for child in node.children:
                    graph.add_edge(node.feature.feature_index, child.feature.feature_index)
                    iterate(child)

        iterate(self.tree)

        graph.write('ademo.dot')

    def predict(self, x):
        """
        Predict the label of a features vector x
        :param x: the features vector x
        :return: the predicted label
        """
        if not self.tree:
            raise RuntimeError("Please train the tree first")

        def walk(node, x):
            """
            Walk in the node until the node is a leaf. Then returns the corresponding label.
            :param node: the node to start with
            :param x: the features vector for which you want to predict the value.
            """
            if node.is_leaf:
                return node.label
            # This is a limitation of the actual code: you can only predict when all the features have already
            # been seen in the training set.
            elif node.feature.extract(x) not in node.feature.outcomes:
                raise RuntimeError("The feature vector contains a value that is not known in the tree")
            else:
                # We get the children node corresponding to the outcome, and we continue walking from there
                node_for_outcome = node.children[node.feature.extract(x)]
                return walk(node_for_outcome, x)

        return walk(self.tree, x)


if __name__ == '__main__':
    clf = DecisionTreeClassifier()
    # Training feature vectors
    x = [[12, 18, 19, 3, 5], [19, 8, 1, 3, 2], [11, 14, 12, 17, 15], [7, 12, 13, 9, 15], [19, 20, 13, 7, 16],
         [2, 4, 20, 12, 18], [4, 19, 7, 9, 16], [19, 4, 1, 13, 2], [16, 10, 8, 9, 19], [13, 10, 15, 1, 11],
         [7, 2, 9, 11, 18], [2, 4, 15, 20, 7], [17, 18, 5, 11, 15], [12, 16, 6, 8, 18], [15, 5, 11, 7, 10],
         [5, 16, 8, 3, 20], [12, 10, 3, 16, 20], [14, 18, 8, 11, 15], [3, 8, 19, 5, 16], [7, 15, 4, 17, 19],
         [14, 8, 16, 4, 10], [1, 14, 17, 13, 8], [7, 14, 20, 1, 12], [4, 10, 16, 13, 12], [13, 14, 10, 19, 9],
         [10, 17, 15, 5, 19], [15, 4, 1, 13, 18], [1, 7, 13, 19, 6], [20, 2, 9, 14, 13], [5, 16, 2, 4, 8],
         [20, 4, 16, 3, 2], [3, 14, 7, 6, 16], [19, 12, 7, 13, 5], [9, 19, 2, 6, 17], [6, 8, 14, 9, 17],
         [8, 17, 10, 15, 4], [10, 3, 12, 1, 16], [4, 10, 20, 15, 19], [9, 5, 3, 6, 1], [17, 11, 9, 3, 15],
         [7, 2, 20, 13, 11], [5, 1, 13, 16, 6], [18, 12, 20, 15, 19], [4, 18, 2, 20, 12], [2, 16, 10, 11, 20],
         [3, 4, 18, 2, 16], [8, 11, 7, 20, 3], [15, 7, 16, 10, 2], [20, 2, 18, 16, 6], [16, 4, 5, 11, 20],
         [15, 12, 10, 3, 5], [11, 4, 12, 17, 20], [4, 3, 16, 11, 2], [12, 10, 7, 8, 16], [20, 15, 1, 6, 17],
         [11, 17, 3, 5, 16], [14, 4, 12, 15, 11], [19, 16, 6, 14, 10], [14, 13, 8, 7, 9], [3, 8, 2, 17, 10],
         [16, 9, 17, 13, 14], [15, 17, 4, 3, 5], [17, 4, 11, 9, 1], [1, 17, 13, 19, 18], [3, 15, 2, 18, 16],
         [18, 4, 9, 6, 10], [1, 10, 18, 16, 6], [3, 9, 7, 14, 11], [15, 1, 7, 5, 14], [10, 11, 8, 6, 13],
         [16, 11, 6, 12, 10], [20, 17, 1, 7, 10], [17, 16, 3, 20, 7], [9, 19, 14, 17, 7], [20, 19, 15, 17, 14],
         [8, 15, 6, 3, 4], [8, 9, 15, 19, 18], [12, 20, 11, 10, 14], [16, 2, 11, 14, 10], [1, 5, 9, 8, 3],
         [5, 10, 1, 9, 2], [16, 17, 5, 15, 2], [12, 7, 13, 15, 5], [16, 1, 6, 12, 10], [6, 9, 3, 17, 19],
         [10, 15, 14, 17, 6], [5, 4, 18, 3, 8], [6, 18, 3, 20, 1], [4, 3, 16, 18, 12], [10, 16, 9, 15, 20],
         [17, 2, 8, 7, 1], [10, 18, 8, 4, 9], [3, 15, 20, 6, 17], [4, 3, 14, 20, 8], [20, 1, 19, 14, 17],
         [5, 18, 2, 17, 14], [7, 11, 12, 4, 3], [12, 7, 3, 11, 9], [1, 13, 14, 2, 20], [20, 1, 13, 6, 12]]

    # Training labels
    y = ['A', 'B', 'D', 'B', 'C', 'B', 'C', 'C', 'D', 'D', 'D', 'C', 'A', 'A', 'D', 'B', 'C', 'A', 'B', 'D', 'B', 'D',
         'A', 'C', 'C', 'C', 'A', 'D', 'B', 'B', 'A', 'D', 'C', 'C', 'B', 'A', 'C', 'B', 'A', 'D', 'D', 'D', 'A', 'D',
         'D', 'C', 'C', 'D', 'C', 'A', 'B', 'B', 'A', 'A', 'B', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'D', 'B',
         'B', 'B', 'C', 'B', 'A', 'A', 'B', 'C', 'B', 'C', 'D', 'C', 'B', 'D', 'B', 'B', 'A', 'A', 'C', 'C', 'D', 'B',
         'D', 'B', 'C', 'D', 'C', 'D', 'D', 'B', 'C', 'A', 'C', 'B']
    clf.fit(x, y)
    print(clf.predict([16, 2, 5, 1, 2, 3]))
