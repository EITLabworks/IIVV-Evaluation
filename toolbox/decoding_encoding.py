import numpy as np


def trim_spaces(strings):
    return ["None" if len(str(s)) == 1 else str(s).strip() for s in strings]


class Encoder:
    def __init__(self, decoded: list, mode="integer", print_info=True):
        """
        mode :
            - "integer" ... create integer category labels.
            - "ohe" ... one hot encoding creates a vector of the number of categories.
        """
        self.mode = mode
        self.decoded = np.array(trim_spaces(decoded))
        self.print_info = print_info

        self.encode()

    def encode(self):
        mapping = dict()

        if len(self.decoded.shape) == 1:
            labels = np.unique(self.decoded)
        elif len(self.decoded.shape) == 2:
            labels = np.unique(self.decoded, axis=0)

        labels = [str(l) for l in labels]
        labels = trim_spaces(labels)

        if self.mode == "integer":
            for n, label in enumerate(labels):
                if self.print_info:
                    print(f"Label '{label}' -> {n}")
                mapping[label] = int(n)

        elif self.mode == "ohe":
            for n, label in enumerate(labels):
                ls = [0 if i != n else 1 for i in range(len(labels))]
                if self.print_info:
                    print(f"Label '{label}' -> {ls}")
                mapping[label] = ls

        encoded = list()
        for entry in self.decoded:
            encoded.append(mapping[str(entry)])

        self.mapping = mapping
        self.encoded = encoded


def hot_coding(categories, info=True):
    categories = np.asarray(categories)

    if len(categories.shape) == 1:
        cases = np.unique(categories)
        if info:
            print(f"Available cases: {cases}")
            print(f"Coding: -> forward")

        mapping = dict()
        for i, c in enumerate(cases):
            case_he = [0 if i != n else 1 for n in range(len(cases))]
            mapping[c] = case_he
        if info:
            print(mapping)

        coding = list()
        for case in categories:
            coding.append(mapping[case])
        coding = np.array(coding)

        inverse_mapping = {tuple(v): str(k) for k, v in mapping.items()}

        return coding, inverse_mapping

    elif len(categories.shape) == 2:
        cases = np.unique(categories, axis=0)
        print(f"Available cases: {cases}")
        print(
            f"Coding: <- backwards | NUMBERING COULD BE INVERTED DUE TO np.unique()!!!"
        )

        mapping = dict()
        for i, c in enumerate(cases):
            mapping[str(c)] = i
        if info:
            print(mapping)

        coding = list()
        for case in categories:
            coding.append(mapping[str(case)])
        coding = np.array(coding)

        inverse_mapping = {tuple(v): str(k) for k, v in mapping.items()}

        return coding, inverse_mapping
    else:
        raise ValueError(f"Invalid input option: {categories}")
