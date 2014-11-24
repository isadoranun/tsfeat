{%- extends 'basic.tpl' -%}
{% from 'mathjax.tpl' import mathjax %}


{%- block header -%}
<!DOCTYPE html>
<html>
<head>

<meta charset="utf-8" />
<title>{{resources['metadata']['name']}}</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>
<script src="toc.js"></script>
<script type="text/javascript">
  $(function() {
    table_of_contents();
});
</script>
{% for css in resources.inlining.css -%}
    <style type="text/css">
    {{ css }}
    </style>
{% endfor %}

<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}

@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

<!-- Loading mathjax macro -->
{{ mathjax() }}

</head>
{%- endblock header -%}

{% block body %}
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">
      <!-- extracted from https://gist.github.com/magican/5574556 -->
<div id="toc-wrapper">
    <div class="header">Contents <a href="#" class="hide-btn">[hide]</a></div>
    <div id="toc"></div>
</div>
 
<style>
  #toc {
    overflow-y: scroll;
    max-height: 300px;
  }
  #toc-wrapper {
    position: fixed; top: 120px; max-width:430px; right: 20px;
    border: thin solid rgba(0, 0, 0, 0.38); opacity: .8;
    border-radius: 5px; background-color: #fff; padding:10px;
    z-index: 100;
  }
  #toc-wrapper.closed {
      min-width: 100px;
      width: auto;
      transition: width;
  }
  #toc-wrapper:hover{
      opacity:1;
  }
  #toc-wrapper .header {
      font-size:18px; font-weight: bold;
  }
  #toc-wrapper .hide-btn {
      font-size: 14px;
  }
 
</style>

<style>
  ol.nested {
    counter-reset: item;
    list-style: none;
  }
  li.nested {
        display: block;
    }
  li.nested:before {
        counter-increment: item;
        content: counters(item, ".")" ";
    }
</style>

{{ super() }}
    </div>
  </div>
</body>
{%- endblock body %}

{% block footer %}
</html>
{% endblock footer %}
