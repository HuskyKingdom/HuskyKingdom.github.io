---
layout: page
permalink: /publications/
title: Publications
description: Publications by Yuhang Song (宋宇航) at the University of Liverpool and National Tsing Hua University, on embodied AI, vision-and-language navigation and robot learning.
years: [2026, 2025, 2024]
nav: false
nav_order: 1
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f {{ site.scholar.bibliography }} -q @*[year={{y}}]* %}
{% endfor %}

</div>
