import math
import operator
from collections import Counter


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.children = []

    def entropy(self):
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

        counter = Counter()
        for training_example_label in self.y:
            counter.update(training_example_label)
        len_dataset = float(len(self.x))
        probability_vector = [count / len_dataset for count in counter.values()]
        return entropy_of_vector(probability_vector)

    def generate_children_for_feature(self, feature):
        children = []
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
            children.append(outcome_node)
        return children

    def choose_best_feature(self, available_features_indices):
        def build_possible_features():
            features = []
            for feature_index in available_features_indices:
                feature = Feature(feature_index)
                for training_example in self.x:
                    feature.add_outcome(training_example[feature_index])
                features.append(feature)
            return features

        features = build_possible_features()
        features_gain = {}
        for feature in features:
            children = self.generate_children_for_feature(feature)
            gain = self.information_gain(children)
            features_gain[feature] = gain
        return max(features_gain.items(), key=operator.itemgetter(1))[0]

    def information_gain(self, children):
        entropy = self.entropy()
        children_weighted_entropy = sum([(len(child.x) / len(self.x)) * child.entropy() for child in children])
        return entropy - children_weighted_entropy

    def is_leaf(self):
        """
        Return true if this is a leaf node ((if the node has a dataset of maximum 1 element, or if all the elements
        have the same label
        """
        return len(self.y) <= 1 or self.y.count(self.y[0]) == len(self.y)


class Feature:
    def __init__(self, feature_index):
        self.feature_index = feature_index
        self.outcomes = set()

    def add_outcome(self, outcome):
        self.outcomes.add(outcome)


class DecisionTree:
    def __init__(self, max_depth=10):
        self.tree = None
        self.available_features = None
        self.max_depth = max_depth

    def fit(self, x, y):
        self.available_features = set(range(len(x[0])))
        root = Node(x, y)  # We create the root node containing all the data

        def expand_node_breadth_first(node, current_depth):
            if node.is_leaf() or current_depth >= self.max_depth:
                return
            best_feature = node.choose_best_feature(self.available_features)
            self.available_features -= set([best_feature.feature_index])
            node.children = node.generate_children_for_feature(best_feature)
            for child in node.children:
                expand_node_breadth_first(child, current_depth + 1)

        expand_node_breadth_first(root, 0)
        self.tree = root

    def predict(self, x):
        pass


if __name__ == '__main__':
    clf = DecisionTree()
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
    pass
