---
title: test shortcode
summary: This is a shortcode test
date: 2024-06-12
authors:
  - admin
tags:
  - Formular

image:
  caption: 'Image credit: [**Unsplash**](https://unsplash.com)'
---


## document 支持 latex 公式渲染

主要包括内联表示和表达式块:

内联表示：

```
{{< math >}}
$\varphi = \dfrac{1+\sqrt5}{2}= 1.6180339887…$
{{< /math >}}
```

具体效果：
{{< math >}}
$\varphi = \dfrac{1+\sqrt5}{2}= 1.6180339887…$
{{< /math >}}



表达式块：
```

{{< math >}}
$$
\varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } }
$$
{{< /math >}}

```

具体效果：
{{< math >}}
$$
 \varphi = 1+\frac{1} {1+\frac{1} {1+\frac{1} {1+\cdots} } }
$$
{{< /math >}}

一组公式

{{< math >}}
$$
    c^Q_t = W^{DQ} h_t \newline
    [q^C_{t,1}; q^C_{t,2}; \ldots; q^C_{t,n}] = q^C_t = W^{UQ} c^Q_t
$$
{{< /math >}}

## bilibili 视频

{{< bilibili id="BV1os411D7be" >}}


## Youtobe 视频

{{< youtube w7Ft2ymGmfc >}}