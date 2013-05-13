---
title: Home
layout: default
---

<h1>Welcome to VSM</h1>
<p>Tutorials for the Vector Space Model Framework developed for InPhO.</p>

<h2>Tutorials</h2>
<ul id="archive">
{% for post in site.posts %}
	<li><a href="{{ post.url }}">{{ post.title }}</a> ({{ post.date | date_to_string }})</li>
{% endfor %}
</ul>

<h2>Authors and Contributors</h2>
<ul>
	<li>Robert Rose (<a href="https://github.com/rrose1" class="user-mention">@rrose1</a>)</li>
	<li>Jun Otsuka (<a href="https://github.com/junotk" class="user-mention">@junotk</a>)</li>
	<li>Tim Downey (<a href="https://github.com/tcdowney" class="user-mention">@tcdowney</a>)</li>
</ul>

<h2>Support</h2>
<p><a href="https://inpho.cogs.indiana.edu/about/">Contact the InPhOrmers</a></p>