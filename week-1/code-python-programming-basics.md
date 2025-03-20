# Python programming basics

## String

### `str.upper()`, `str.lower()`, `str.title()`

Fill `____` pieces below to have correct values for `lower_cased`, `upper_cased` and `title_cased` variables.


```python
original = 'Python strings are COOL!'
```


```python
# Your solution here:
lower_cased = original.____
upper_cased = ____.upper()
title_cased = ____.____
```




    'Python Strings Are Cool!'




```python
# Let's verify that the implementation is correct by running the cell below
# `assert` will raise `AssertionError` if the statement is not true.
assert lower_cased == 'python strings are cool!'
assert upper_cased == 'PYTHON STRINGS ARE COOL!'
assert title_cased == 'Python Strings Are Cool!'
```

### `str.replace()`


```python
my_string = 'Python is my favorite programming language!'
```


```python
# Your solution here
____
```


```python
assert my_modified_string == 'Python will be my favorite programming language!'
```

### `str.format()`


```python
secret = '____ is cool.'.format(____)
```


```python
assert secret == 'Python is cool.'
```

### `str.join()`


```python
pandas = 'pandas'
numpy = 'numpy'
requests = 'requests'
```


```python
# Your solution here:
cool_python_libs = ____
```


```python
assert cool_python_libs == 'pandas, numpy, requests'
```

### `str.strip()`


```python
ugly_formatted = ' \n \t Some story to tell '
```


```python
# Your solution:
stripped = ____
```


```python
assert stripped == 'Some story to tell'
```

### `str.split()`


```python
sentence = 'three different words'
```


```python
# Your solution:
words = ____
```


```python
assert words == ['three', 'different', 'words']
```

### `\n`, `\t`


```python
# Your solution:
two_lines = 'First line____Second line'
indented = '____This will be indented'
```


```python
assert two_lines == '''First line
Second line'''
assert indented == '	This will be indented'
```

## Numbers

### Creating formulas

Write the following mathematical formula in Python:

$result = 6a^3 - \frac{8b^2 }{4c} + 11$



```python
a = 2
b = 3
c = 2
```


```python
# Your formula here:
result = ____
```


```python

assert result == 50
```

### Floating point pitfalls

Make assertion for `0.1 + 0.2 == 0.3`


```python
# This won't work:
# assert 0.1 + 0.2 == 0.3

# Your solution here:
____
```

### Floor division `//`, modulus `%`, power `**`


```python
assert 7 // 5 == ____
assert 7 % 5 == ____
assert 2 ** 3 == ____ 
```

## Lists

### `list.append()`, `list.remove()`, mutable


```python
# Let's create an empty list.
my_list = ____

# Let's add some values
my_list.____('Python')
my_list.____('is ok')
my_list.____('sometimes')

# Let's remove 'sometimes'
my_list.____('sometimes')

# Let's change the second item
my_list[____] = 'is neat'
```


```python
assert my_list == ['Python', 'is neat']
```

### Slice

Create a new list without modifiying the original one.


```python
original = ['I', 'am', 'learning', 'hacking', 'in']
```


```python
# Your implementation here
modified = ____
```


```python
assert original == ['I', 'am', 'learning', 'hacking', 'in']
assert modified == ['I', 'am', 'learning', 'lists', 'in', 'Python']
```

### `list.extend()`


```python
first_list = ['beef', 'ham']
second_list = ['potatoes', 1, 3]
```


```python
# Your solution:
# use `extend()
merged_list = ____
```


```python
assert merged_list == ['beef', 'ham', 'potatoes', 1]
```


```python
third_list = ['beef', 'ham']
forth_list = ['potatoes', 1, 3]
```


```python
# Your soultion:
# use `+` operator
merged_list = ____
```


```python
assert merged_list == ['beef', 'ham', 'potatoes', 1]
```

### `list.sort()`

Create a merged sorted list.


```python
my_list = [6, 12, 5]
```


```python
# Your implementation here
____
```


```python
assert my_list == [12, 6, 5]
```

### `sorted(list)`


```python
numbers = [8, 1, 6, 5, 10]
```


```python
sorted_numbers = ____
```


```python
assert sorted_numbers == [1, 5, 6, 8, 10]
```

### `list.reverse()`


```python
my_list = ['c', 'b', 'ham']
```


```python
# Your solution:
____
```


```python
assert my_list == ['ham', 'b', 'c']
```

## Dictionaries

### Populating a dictionary

Create a dictionary by using all the given variables.


```python
first_name = 'John'
last_name = 'Doe'
favorite_hobby = 'Python'
sports_hobby = 'gym'
age = 82
```


```python
# Your implementation
my_dict = ____
```


```python
assert my_dict == {
        'name': 'John Doe',
        'age': 82,
        'hobbies': ['Python', 'gym']
    }
```

### `del`


```python
my_dict = {'key1': 'value1', 'key2': 99, 'keyX': 'valueX'}
key_to_delete = 'keyX'
```


```python
# Your solution here:
if key_to_delete in my_dict:
    ____
```


```python
assert my_dict == {'key1': 'value1', 'key2': 99}
```

### Mutable


```python
my_dict = {'ham': 'good', 'carrot': 'semi good'}
```


```python
# Your solution here:
____
```


```python
assert my_dict['carrot'] == 'super tasty'
```

### `dict.get()`


```python
my_dict = {'a': 1, 'b': 2, 'c': 3}
```


```python
# Your solution here:
d = ____
```


```python
assert d == 'default value'
```


```python
assert my_dict == {'a': 1, 'b': 2, 'c': 3}
```

### `dict.setdefault()`


```python
my_dict = {'a': 1, 'b': 2, 'c': 3}
```


```python
# Your solution here:
d = ____
```


```python
assert d == 'default value'
```


```python
assert my_dict == {'a': 1, 'b': 2, 'c': 3, 'd': 'default value'}
```

### Accessing and merging dictionaries

Combine `dict1`, `dict2`, and `dict3` into `my_dict`. In addition, get the value of `special_key` from `my_dict` into a `special_value` variable. Note that original dictionaries should stay untouched and `special_key` should be removed from `my_dict`.


```python
dict1 = dict(key1='This is not that hard', key2='Python is still cool')
dict2 = {'key1': 123, 'special_key': 'secret'}
# This is also a away to initialize a dict (list of tuples) 
dict3 = dict([('key2', 456), ('keyX', 'X')])
```


```python
# Your impelementation
my_dict = ____
special_value = ____
```


```python
assert my_dict == {'key1': 123, 'key2': 456, 'keyX': 'X'}
assert special_value == 'secret'

# Let's check that the originals are untouched
assert dict1 == {
        'key1': 'This is not that hard',
        'key2': 'Python is still cool'
    }
assert dict2 == {'key1': 123, 'special_key': 'secret'}
assert dict3 == {'key2': 456, 'keyX': 'X'}
```

## Acknowledgments

Thanks to below awesome open source projects for Python learning, which inspire this chapter.

- [learn-python](https://github.com/trekhleb/learn-python) and [Oleksii Trekhleb](https://github.com/trekhleb)
- [ultimate-python](https://github.com/huangsam/ultimate-python) and [Samuel Huang](https://github.com/huangsam)
- [learn-python3](https://github.com/jerry-git/learn-python3) and [Jerry Pussine](https://github.com/jerry-gitq )
