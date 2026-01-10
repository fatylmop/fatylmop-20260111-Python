```go
package main

import (
  "fmt"
    "math"
      "strings"
      )

      func main() {
        // 定义一个字符串
          s := "Hello, World!"

            // 打印字符串
              fmt.Println(s)

                // 切片操作
                  slice := []int{1, 2, 3, 4, 5}
                    fmt.Println(slice)

                      // 数组操作
                        array := [5]int{1, 2, 3, 4, 5}
                          fmt.Println(array)

                            // 字符串转大写
                            .upper := strings.ToUpper(s)
                              fmt.Println(.upper)

                                // 计算圆周率
                                  pi := math.Pi
                                    fmt.Println(pi)

                                      // 判断奇偶数
                                        fmt.Println("Is 2 even?", 2%2 == 0)
                                          fmt.Println("Is 3 even?", 3%2 == 0)

                                            // 循环输出数字
                                              for i := 0; i < 5; i++ {
                                                  fmt.Println(i)
                                                    }

                                                      // 切片添加元素
                                                        slice = append(slice, 6)
                                                          fmt.Println(slice)

                                                            // 使用map
                                                              m := make(map[string]int)
                                                                m["one"] = 1
                                                                  m["two"] = 2
                                                                    fmt.Println(m)

                                                                      // 删除map中的键值对
                                                                        delete(m, "one")
                                                                          fmt.Println(m)

                                                                            // 结构体定义
                                                                              type Point struct {
                                                                                  X, Y int
                                                                                    }
                                                                                      p := Point{1, 2}
                                                                                        fmt.Println(p)

                                                                                          // 指针操作
                                                                                            q := &p
                                                                                              fmt.Println(*q)

                                                                                                // 方法调用
                                                                                                  fmt.Println(p.X, p.Y)

                                                                                                    // 函数定义
                                                                                                      f := func(x int) int {
                                                                                                          return x * x
                                                                                                            }
                                                                                                              fmt.Println(f(4))
                                                                                                              
                                                                                                                // 指针作为参数
                                                                                                                  fmt.Println(f(*q))
                                                                                                                  
                                                                                                                    // defer语句
                                                                                                                      defer fmt.Println("Deferred print")
                                                                                                                        fmt.Println("Immediate print")
                                                                                                                        }
                                                                                                                        ```
