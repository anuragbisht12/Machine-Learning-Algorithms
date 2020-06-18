from fractions import Fraction
import os
import unittest
from sentiment_classifier import SentimentClassifier
class TestSentimentClassifier(unittest.TestCase):
    def setUp(self):
        pass

    def test_validate(self):
        """cross validates with an error of 35% or less"""

        neg = self.split_file('data/rt-polaritydata/rt-polarity.neg')
        pos = self.split_file('data/rt-polaritydata/rt-polarity.pos')
        classifier = SentimentClassifier.build([
        neg['training'],
        pos['training']
        ])
        c = 2 ** 7
        classifier.c = c
        classifier.reset_model()
        n_er = self.validate(classifier, neg['validation'], 'negative')
        p_er = self.validate(classifier, pos['validation'], 'positive')
        total = Fraction(n_er.numerator + p_er.numerator,
        n_er.denominator + p_er.denominator)
        print(total)
        self.assertLess(total, 0.35)
    
    def test_validate_itself(self):
        """yields a zero error when it uses itself"""

        classifier = SentimentClassifier.build([
        'data/rt-polaritydata/rt-polarity.neg',
        'data/rt-polaritydata/rt-polarity.pos'
        ])
        c = 2 ** 7
        classifier.c = c
        classifier.reset_model()
        n_er = self.validate(classifier,
        'data/rt-polaritydata/rt-polarity.neg',
        'negative')
        p_er = self.validate(classifier,
        'data/rt-polaritydata/rt-polarity.pos',
        'positive')
        total = Fraction(n_er.numerator + p_er.numerator,
        n_er.denominator + p_er.denominator)
        print(total)
        self.assertEqual(total, 0)