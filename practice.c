#include<stdio.h>

#define MAX_CHILDREN 10

int main() {
    typedef struct MultiTreeNode {
        int data;
        int childCount;
        struct MultiTreeNode* children[MAX_CHILDREN];
    } MultiTreeNode;
    return 0;
}