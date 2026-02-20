


















import pytest
import pandas as pd
import numpy as np
from preprocessing.dataset_validator import (
    ReadinessIssue,
    ReadinessReport,
    DatasetReadinessValidator
)


class TestReadinessIssue:



    def test_create_critical_issue(self):

        issue = ReadinessIssue(
            level="CRITICAL",
            code="CAT_DOM",
            message="Too many categorical columns",
            impact="Expected 0% accuracy",
            recommendation="Convert to numeric"
        )

        assert issue.level == "CRITICAL"
        assert issue.code == "CAT_DOM"
        assert "categorical" in issue.message.lower()

    def test_create_warning_issue(self):

        issue = ReadinessIssue(
            level="WARNING",
            code="NUM_STR",
            message="Numeric as strings",
            impact="Won't be discretized",
            recommendation="Convert to numeric"
        )

        assert issue.level == "WARNING"

    def test_create_info_issue(self):

        issue = ReadinessIssue(
            level="INFO",
            code="MISS_INFO",
            message="Missing values detected",
            impact="Will be imputed",
            recommendation="Automatic"
        )

        assert issue.level == "INFO"


class TestReadinessReport:



    def test_create_empty_report(self):

        report = ReadinessReport(score=100)

        assert report.score == 100
        assert len(report.issues) == 0
        assert len(report.passed_checks) == 0
        assert report.verdict == "RECOMMENDED"

    def test_get_critical_issues(self):

        report = ReadinessReport(
            score=0,
            issues=[
                ReadinessIssue("CRITICAL", "C1", "c1", "i1", "r1"),
                ReadinessIssue("WARNING", "W1", "w1", "i2", "r2"),
                ReadinessIssue("CRITICAL", "C2", "c2", "i3", "r3"),
            ]
        )

        critical = report.get_critical_issues()
        assert len(critical) == 2
        assert all(i.level == "CRITICAL" for i in critical)

    def test_get_warning_issues(self):

        report = ReadinessReport(
            score=50,
            issues=[
                ReadinessIssue("WARNING", "W1", "w1", "i1", "r1"),
                ReadinessIssue("INFO", "I1", "i1", "i2", "r2"),
                ReadinessIssue("WARNING", "W2", "w2", "i3", "r3"),
            ]
        )

        warnings = report.get_warning_issues()
        assert len(warnings) == 2

    def test_get_info_issues(self):

        report = ReadinessReport(
            score=90,
            issues=[
                ReadinessIssue("INFO", "I1", "i1", "i1", "r1"),
                ReadinessIssue("WARNING", "W1", "w1", "i2", "r2"),
            ]
        )

        infos = report.get_info_issues()
        assert len(infos) == 1
        assert infos[0].level == "INFO"


class TestCategoricalDominance:



    def test_critical_over_50_percent_categorical(self):

        df = pd.DataFrame({
            'cat1': ['a', 'b', 'c'] * 5,
            'cat2': ['x', 'y', 'z'] * 5,
            'cat3': ['p', 'q', 'r'] * 5,
            'num1': [1, 2, 3] * 5,
            'class': ['A', 'B', 'A'] * 5
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')


        assert report.score == 0
        assert report.verdict == "NOT_RECOMMENDED"
        critical = report.get_critical_issues()
        assert len(critical) == 1
        assert critical[0].code == "CAT_DOM"

    def test_warning_30_to_50_percent_categorical(self):

        df = pd.DataFrame({
            'cat1': ['a', 'b', 'c'] * 50,
            'cat2': ['x', 'y', 'z'] * 50,
            'num1': [1, 2, 3] * 50,
            'num2': [4, 5, 6] * 50,
            'num3': [7, 8, 9] * 50,
            'class': ['A', 'B', 'A'] * 50
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')


        assert report.score == 70
        assert report.verdict == "CAUTION"
        warnings = report.get_warning_issues()
        assert any(w.code == "CAT_WARN" for w in warnings)

    def test_ok_less_than_30_percent_categorical(self):

        df = pd.DataFrame({
            'cat1': ['a', 'b', 'c'] * 5,
            'num1': [1, 2, 3] * 5,
            'num2': [4, 5, 6] * 5,
            'num3': [7, 8, 9] * 5,
            'num4': [10, 11, 12] * 5,
            'class': ['A', 'B', 'A'] * 5
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')


        assert report.score >= 80
        passed_text = ' '.join(report.passed_checks)
        assert 'acceptable' in passed_text.lower() or 'categorical' in passed_text.lower()

    def test_fully_numeric_dataset(self):

        df = pd.DataFrame({
            'num1': [1, 2, 3] * 5,
            'num2': [4, 5, 6] * 5,
            'num3': [7, 8, 9] * 5,
            'class': ['A', 'B', 'A'] * 5
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        assert report.score >= 90
        assert report.verdict == "RECOMMENDED"
        passed_text = ' '.join(report.passed_checks)
        assert 'numeric' in passed_text.lower()


class TestNumericAsStrings:



    def test_detects_numeric_as_string_with_missing_markers(self):

        df = pd.DataFrame({
            'col1': ['1.0', '2.0', '3.0', '?', '5.0'],
            'col2': [1, 2, 3, 4, 5],
            'class': ['A', 'B', 'A', 'B', 'A']
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        warnings = report.get_warning_issues()
        num_str_warnings = [w for w in warnings if w.code == "NUM_STR"]
        assert len(num_str_warnings) == 1
        assert 'col1' in num_str_warnings[0].message

    def test_no_warning_for_true_categorical(self):

        df = pd.DataFrame({
            'cat1': ['red', 'blue', 'green', 'red', 'blue'],
            'num1': [1, 2, 3, 4, 5],
            'class': ['A', 'B', 'A', 'B', 'A']
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        warnings = report.get_warning_issues()
        num_str_warnings = [w for w in warnings if w.code == "NUM_STR"]
        assert len(num_str_warnings) == 0

    def test_detects_multiple_numeric_as_string_columns(self):

        df = pd.DataFrame({
            'col1': ['1', '2', '3'],
            'col2': ['4.5', '5.5', '6.5'],
            'num1': [1, 2, 3],
            'class': ['A', 'B', 'A']
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        warnings = report.get_warning_issues()
        num_str_warnings = [w for w in warnings if w.code == "NUM_STR"]
        assert len(num_str_warnings) == 2


class TestDatasetSize:



    def test_critical_below_min_samples(self):

        df = pd.DataFrame({
            'num1': [1, 2, 3],
            'class': ['A', 'B', 'A']
        })

        validator = DatasetReadinessValidator(min_samples=10)
        report = validator.validate(df, 'class')


        assert report.score == 0
        assert report.verdict == "NOT_RECOMMENDED"
        critical = report.get_critical_issues()
        assert any(c.code == "SIZE_MIN" for c in critical)

    def test_warning_below_100_samples(self):

        df = pd.DataFrame({
            'num1': range(50),
            'class': ['A', 'B'] * 25
        })

        validator = DatasetReadinessValidator(min_samples=10)
        report = validator.validate(df, 'class')


        warnings = report.get_warning_issues()
        assert any(w.code == "SIZE_SMALL" for w in warnings)

    def test_ok_above_100_samples(self):

        df = pd.DataFrame({
            'num1': range(150),
            'class': ['A', 'B'] * 75
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')


        passed_text = ' '.join(report.passed_checks)
        assert 'size' in passed_text.lower() or '150' in passed_text


class TestMissingValues:



    def test_warning_missing_in_decision_column(self):

        df = pd.DataFrame({
            'num1': [1, 2, 3, 4],
            'class': ['A', 'B', None, 'A']
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        warnings = report.get_warning_issues()
        assert any(w.code == "MISS_DEC" for w in warnings)

    def test_info_missing_in_features(self):

        df = pd.DataFrame({
            'num1': [1, np.nan, 3, 4],
            'num2': [5, 6, np.nan, 8],
            'class': ['A', 'B', 'A', 'B']
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        infos = report.get_info_issues()
        assert any(i.code == "MISS_INFO" for i in infos)

    def test_passed_no_missing_values(self):

        df = pd.DataFrame({
            'num1': [1, 2, 3, 4],
            'num2': [5, 6, 7, 8],
            'class': ['A', 'B', 'A', 'B']
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        passed_text = ' '.join(report.passed_checks)
        assert 'missing' in passed_text.lower()


class TestClassBalance:



    def test_critical_only_one_class(self):

        df = pd.DataFrame({
            'num1': [1, 2, 3, 4],
            'class': ['A', 'A', 'A', 'A']
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        assert report.score == 0
        critical = report.get_critical_issues()
        assert any(c.code == "BAL_ONE" for c in critical)

    def test_warning_imbalanced_over_10x(self):

        df = pd.DataFrame({
            'num1': range(100),
            'class': ['A'] * 95 + ['B'] * 5
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        warnings = report.get_warning_issues()
        assert any(w.code == "BAL_IMB" for w in warnings)

    def test_ok_balanced_classes(self):

        df = pd.DataFrame({
            'num1': range(100),
            'class': ['A', 'B'] * 50
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        passed_text = ' '.join(report.passed_checks)
        assert 'balance' in passed_text.lower() or 'ratio' in passed_text.lower()


class TestConstantColumns:



    def test_info_constant_column_detected(self):

        df = pd.DataFrame({
            'const1': [5, 5, 5, 5],
            'num1': [1, 2, 3, 4],
            'class': ['A', 'B', 'A', 'B']
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        infos = report.get_info_issues()
        assert any(i.code == "CONST_COL" for i in infos)
        const_issue = [i for i in infos if i.code == "CONST_COL"][0]
        assert 'const1' in const_issue.message

    def test_passed_no_constant_columns(self):

        df = pd.DataFrame({
            'num1': [1, 2, 3, 4],
            'num2': [5, 6, 7, 8],
            'class': ['A', 'B', 'A', 'B']
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        passed_text = ' '.join(report.passed_checks)
        assert 'constant' in passed_text.lower()


class TestScoringAndVerdict:



    def test_verdict_recommended_score_80_to_100(self):

        df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5] * 30,
            'num2': [6, 7, 8, 9, 10] * 30,
            'class': ['A', 'B'] * 75
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        assert report.score >= 80
        assert report.verdict == "RECOMMENDED"

    def test_verdict_caution_score_50_to_79(self):

        df = pd.DataFrame({
            'cat1': ['a', 'b', 'c'] * 50,
            'cat2': ['x', 'y', 'z'] * 50,
            'num1': [1, 2, 3] * 50,
            'num2': [4, 5, 6] * 50,
            'num3': [7, 8, 9] * 50,
            'class': ['A', 'B'] * 75
        })


        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        assert 50 <= report.score < 80
        assert report.verdict == "CAUTION"

    def test_verdict_not_recommended_score_0_to_49(self):

        df = pd.DataFrame({
            'cat1': ['a', 'b', 'c'] * 50,
            'cat2': ['x', 'y', 'z'] * 50,
            'cat3': ['p', 'q', 'r'] * 50,
            'num1': [1, 2, 3] * 50,
            'class': ['A', 'B'] * 75
        })


        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        assert report.score < 50
        assert report.verdict == "NOT_RECOMMENDED"


class TestIntegration:



    def test_perfect_dataset(self):

        df = pd.DataFrame({
            'num1': range(200),
            'num2': range(200, 400),
            'num3': range(400, 600),
            'class': ['A', 'B', 'C', 'D'] * 50
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')

        assert report.score == 100
        assert report.verdict == "RECOMMENDED"
        assert len(report.get_critical_issues()) == 0
        assert len(report.passed_checks) > 0

    def test_problematic_dataset(self):

        df = pd.DataFrame({
            'cat1': ['a', 'b', 'c', 'd', 'e'],
            'cat2': ['x', 'y', 'z', 'w', 'v'],
            'cat3': ['p', 'q', 'r', 's', 't'],
            'class': ['A', 'A', 'A', 'A', 'B']
        })

        validator = DatasetReadinessValidator(min_samples=10)
        report = validator.validate(df, 'class')





        assert report.score == 0
        assert report.verdict == "NOT_RECOMMENDED"
        assert len(report.get_critical_issues()) >= 1

    def test_decision_column_not_found(self):

        df = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [4, 5, 6]
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'nonexistent_column')

        assert report.score == 0
        assert report.verdict == "NOT_RECOMMENDED"
        critical = report.get_critical_issues()
        assert len(critical) == 1
        assert critical[0].code == "DC_NOT_FOUND"

    def test_print_report_does_not_crash(self):

        df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5] * 30,
            'class': ['A', 'B'] * 75
        })

        validator = DatasetReadinessValidator()
        report = validator.validate(df, 'class')


        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured
        report.print_report()
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "DATASET READINESS REPORT" in output
        assert "Score:" in output
