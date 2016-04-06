import math
from collections import Counter, deque


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
                if outcome == training_example_features[feature.feature_index]:
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

    @property
    def label(self):
        """
        Return the most common label in the node's dataset.
        """
        if not self.y:
            raise RuntimeError("The node has an empty dataset")  # Should not happen
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
    x = [
        [1, 2, 3, 1, 2, 1],
        [1, 4, 3, 1, 2, 3],
        [5, 1, 3, 3, 2, 5],
        [1, 2, 3, 1, 3, 1],
        [1, 2, 3, 1, 2, 3],
        [1, 2, 5, 1, 5, 1],
        [5, 2, 8, 5, 2, 5],
        [5, 4, 8, 5, 2, 8],
        [5, 5, 8, 8, 2, 5],
        [5, 2, 8, 3, 8, 5],
        [5, 2, 8, 5, 2, 8],
        [5, 2, 5, 5, 5, 5],
    ]
    # Training labels
    y = [
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "B",
        "B",
        "B",
        "B",
        "B",
        "B",
    ]
    clf.fit(x, y)  # "Learning" step
    print(clf.predict([1, 2, 5, 1, 2, 3]))
