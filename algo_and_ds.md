# Runestone Academy CS + DS

## Computer science is the study of algorithms
[link]()
* Abstraction
    - Logical (how things are used, interface between it end user)
    - Physical (how things actually work, and take place under the hood)
* Procedural Abstraction
    - Black Box Thinking, using a function
* Data Abstraction
    - Abstract Data Typer (ADT) is an encapsulation of a data type
    - Provides an implementation-independent view of data

* Intractable - no solution
* Computable - solution exists

Error Handling
* Syntax error
    - missing a colon, paraenth
    - python cannot interpret the code written
* Logic error
    - gives wrong result
    - results in an exception
    - can be caught, with try/except or raised explicitly with raise

Thought problem
* Infinite Monkey Theorem
    - if a monkey typed randomly on a keyboard long enough, it will produce shakespeare

Inheritence
* Inheritance Heirarchy:
    - Shows the relationships between objects
    - Subclasses
    - Superclasses
* IS-A
    - Describes that a child class is related to a parent class
    - Includes inheretance
* HAS-A
    - Doesn't include inheretance
    - Are outside of the inheretance hierarchy
    - Have instances of the heirarchy within them

## Design Patterns
[link](https://www.toptal.com/python/python-design-patterns)

Gang of Four (GOF)
* Program to an interface not an implementation
    - meaning program to the type of object you expect
* Favor object composition over inheritance
    - keep the number of objects to a min

Behavioral patterns
* Chain of responsibility
    - create a small chunk of code for a single thing
    - single responsibility principle
* Command
    - create blocks that have the same methods
    - tie them to a list and then grant ability to 'method' each, like undo
* Injection
    - configure objects within config and pass to everything else
    - as opposed to instantiating them within each use
* many more...

Strctural patterns
* Facade
    - a class that groups other classes as components under within it
* Interface
    - modify an existing heavily tested code with a different face
* Decorator
    -

## Searching and Sorting

* Sequential Search o(n)
    - exhaustive scan through all items in a list
    - item is present
        - best case is o(1)
        - worst case is o(n)
        - avg case is o(n/2)
    - item is not present
        - best case is o(n)
        - worst case is o(n)
        - avg case is o(n)

* Sequential Search with Sorting o(n/2)
    - allows you to stop looking after you go past the item's value reducing runtime
    - item is not present
        - best case is o(n) if sorted o(1)
        - worst case is o(n) if sorted o(n)
        - avg case is o(n) if sorted o(n/2)

* Binary Search o(log(n))
    - divide and conquer strategy
        - break problem into smaller subproblems (recursive)
    - searching an ordered list
    - start by comparing he middle item
    - if it is not correct, we can eliminate half the values by the ordered nature of the list

* Sorting
    - for small collections a complex sorting method may be more trouble than is worth
    - evaluated by two types of operations:
        - number of comparisons made
        - number of exchanges made bewteen items in a list

* Exchanging or Swapping
    - typically needs a temp val in other languages
        - t = l[i]
        - l[i] = l[j]
        - l[j] = t
    - in python its
        - a,b = b,a

* Bubble Sort
    - makes multiple passes through a list
    - it compares adjacent items and exchanges ones that are out of order
    - most inefficient sort
    
* Short Bubble
    - includes early stopping once the list has no exchanges at a given pass

* Selection Sort
    - improves on bubble sort by making fewer exchanges
    - ultimately just finds max and then puts it in the right spot

* Insertion Sort
    - creates a sorted list at the beginning and then inserts new values within the sorted sublist

* Shell Sort
    - is an insertion sort where there are n sublists being sorted at once
    - on each pass a smaller step can be used that gets closer to a standard insert sort
    - finally on the last pass a normal insertion sort is used

## Hashing

* Hashing
    - a hash table is a collection of items which are stored to make it easy to find them later
    - each position is called a slot
    - the hash_function is such that it creates a unique map between an item and a slot. so that a slot cannot have more than 2 values
    - if this occurs its called a collision
    - if we know all items and are sure they will never change, then it is possible to create a perfect hash function
    - of course you can always increase the hash_function to handle the full possiblility of numbers. however that measn that you'll be creating a list of len(n) with Nones for the vast majority wasting a large amount of memory
    - best hash function is one that reduces the number of collisions, is easy to compute, and evenly distributes the items within the hash table

* Remainder Method
    - hash_function hf(i) = i%len(t) where t is the table

* Folding Method
    - chops item into x parts, adds parts then % len(t)
    - some reverse every other part of x

* Mid-square Method
    - square the value, take the middle values and then % len(t)

* Collision Resolution (rehashing)
    - the table size is often set to a prime number to ensure that all slots in the table will be visited. meaning it doesn't matter what plus x probing method is used
    - open addressing
        - looking for unused slots to fill the item that caused a collision
        - puts the collided item in the next availble slot
    - linear probing
        - process of looking for collided values
        - if the hash value isn't the item value, then you look sequentially through the next values until you find the item or you find a None (since you know the item would have filled it if it were placed into the hash table you know its not there)
    - plus 3 probing
        - addresses clustering of values around highly occuring collisions
        - instead of seqentialy filling Nones, it uses every 3rd slot until a None is found
    - quadratic probing
        - uses a skip of sucessive perfect squares
    - chaining
        - allows collided items to become a chain off a single hash slot

* Map (Dict) Data Type
    - two lists, one that holds slots the other that counts as the index lookup

* Load Factor
    - lambda is the load factor
    - it represents how filled the hash table is
    - successful search .5(1+1/(1-lambda)))
    - unsuccessful search .5(1+(1/(1-lambda))^2)



## LeetCode

Example patterns to solve problems:
    -
    - https://medium.com/leetcode-patterns/leetcode-pattern-2-sliding-windows-for-strings-e19af105316b
