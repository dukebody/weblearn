import pytest
import numpy as np

from base import AbstractModel, ValuesModel, KeyValueModel, Validator


def aequal(a1, a2):
    """
    Return True if numpy arrays a1 and a2 are equal.
    """
    return (a1 == a2).all()


# AbstractModel
def test_to_array():
    am = AbstractModel()
    array = am.to_array(['1', '3'])
    assert aequal(array, np.array([1, 3]))


def test_to_array_nofloats():
    am = AbstractModel()
    with pytest.raises(ValueError):
        am.to_array(['1', '3b'])


# ValuesModel
def test_vm_form_to_list():
    vm = ValuesModel()
    vars = vm.form_to_list({'values': '1,2'})
    assert vars == ['1', '2']


def test_vm_form_to_list_strip():
    vm = ValuesModel()
    vars = vm.form_to_list({'values': '1, 2'})
    assert vars == ['1', '2']


def test_vm_form_to_list_missing():
    vm = ValuesModel()
    with pytest.raises(ValueError):
        vm.form_to_list({})


# KeyValueModel
class TestKeyValueModel(KeyValueModel):
    schema = [{'name': 'a'}, {'name': 'b'}]


def test_kvm_variables():
    kvm = TestKeyValueModel()
    vars = kvm.form_to_list({'a': '1', 'b': '2'})
    assert vars == ['1', '2']


def test_kvm_variables_missing():
    kvm = TestKeyValueModel()
    with pytest.raises(ValueError):
        kvm.form_to_list({'a': '1'})


# Validator
def test_schema_presence():
    schema = [
        {'name': 'john'},
        {'name': 'lisa'}
    ]
    validator = Validator(schema)
    assert validator.validate({'john': 's', 'lisa': 'f'})
    assert not validator.validate({'john': 's'})
    assert 'lisa' in validator.errors


def test_schema_default():
    schema = [
        {'name': 'john', 'default': 'a'},
    ]
    validator = Validator(schema)
    assert validator.validate({})
    assert 'john' not in validator.errors
    assert validator.cleaned_data == ['a']


def test_schema_cleaned_data():
    schema = [
        {'name': 'john'},
        {'name': 'lisa'}
    ]
    validator = Validator(schema)
    validator.validate({'john': 's', 'lisa': 'f'})
    assert validator.cleaned_data == ['s', 'f']

    validator.validate({'john': 's'})
    assert validator.cleaned_data is None


def test_schema_transform():
    schema = [
        {'name': 'john', 'transform': int}
    ]
    validator = Validator(schema)
    assert validator.validate({'john': '1'})
    assert validator.cleaned_data == [1]


def test_schema_transform_error():
    def func(value):
        raise Exception('Value could not be transformed')
    schema = [
        {'name': 'john', 'transform': func}
    ]
    validator = Validator(schema)
    assert not validator.validate({'john': 'whatever'})
    assert validator.cleaned_data is None
    assert 'john' in validator.errors