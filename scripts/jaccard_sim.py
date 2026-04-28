# Jaccard similarity function
def calculate_jaccard_similarity(tags1, tags2):
    set1 = set(tags1.split(' | '))
    set2 = set(tags2.split(' | '))
    if not set1 or not set2:  # Check if either set is empty
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

tags1 = 'alcohol | bottle | liquor | display | fill | sake | shelf | shop window | showcase | sign | store | storefront | window | window display | wine bottle | writing'
tags2 = 'alcohol | beer | beer bottle | beverage | bottle | liquor | fill | liquor store | sake | shelf | shop window | store | storefront | window | writing'
r=calculate_jaccard_similarity(tags1, tags2)
print(r)