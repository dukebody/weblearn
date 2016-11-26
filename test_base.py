import pytest
import numpy as np

from base import AbstractModel, ValuesModel, KeyValueModel


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
    fields = ['a', 'b']


def test_kvm_variables():
    kvm = TestKeyValueModel()
    vars = kvm.form_to_list({'a': '1', 'b': '2'})
    assert vars == ['1', '2']


def test_kvm_variables_missing():
    kvm = TestKeyValueModel()
    with pytest.raises(ValueError):
        kvm.form_to_list({'a': '1'})



