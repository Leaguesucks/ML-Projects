#include "gradient_check.h"

int main() {
    dnn::test_gradient_check(4, {3}, 2);
}
