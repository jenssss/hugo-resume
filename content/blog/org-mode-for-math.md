+++
title = "Math derivations with collapsibles"
author = ["Jens Svensmark"]
date = 2020-04-03
tags = ["howto", "emacs", "org-mode", "html"]
draft = false
math = true
#comment = "this file is [auto-generated]"
+++

How to export mathematical derivations to html using [org-mode](https://orgmode.org/), with
collapsibles to help organizing the content.

<!--more-->

Working as a physisict, I often write technical documentation
outlining a mathematical derivation. For this purpose, [LaTeX](https://www.latex-project.org/) is _the_
standard language. With LaTeX, one can create high-quality documents
with beautiful typography and it includes support for practically any
mathematical notation you could want.

The challenge with writing down a mathematical derivation is the
various levels of abstraction that are involved. If I write out all
the little details of how to do each step in the derivation, the
resulting document becomes very long and basically impossible to
navigate. If on the other hand I just show the most important
equations of the derivation, it becomes easier to read and follow the
overall structure of the derivation, but very difficult to redo the
derivation, since all the little details are missing. Ideally I want
both, to include all the little details, but also be able to hide
them, in order to get an overview of the derivation as a whole.

The problem is that the standard way of using LaTeX is to first write
source code in a `.tex` file, then compile it to `.pdf` format. For
writing scientic papers or a thesis, the static pdf format is quite
well suited. But for my own derivations where I wanted to be able to
hide details away, but still have them close at hand, I needed
something more dynamic.

I thought for a while about what would be a suitable alternative to
pdf, until finally I realized the humble internet browser had just the
features I was looking for. I remembered seeing wikipedia articles,
where one could click a button to show and hide parts of a
mathematical derivation. So I started looking into how I could
make similar collapsible derivations in html.


## org-mode {#org-mode}

As an avid user of the emacs editor, I found that [org-mode](https://orgmode.org/) seemed to
fit what I was looking for. Instead of writing files directly in the
`.tex` format, I would write in the `.org` format (a kind of markup),
and then export that to either `.tex` or to `.html`. When editing a
file in org-mode, any section or block can be collapsed to more easily
navgiate the file. Org-mode even came with builtin support for LaTeX
rendering in the `.html` output using the web based LaTeX renderer
[MathJax](https://www.mathjax.org/).

Next I wanted to add collapsible structures to my `.html` output.  The
theme I was using for the html export was based on Bootstrap, so
combining the example in the [bootstrap documentation](https://getbootstrap.com/docs/4.0/components/collapse/#example) and [this org-mode
guide](https://orgmode.org/manual/Advanced-Export-Configuration.html) I added the following to my `~/.emacs` file

```lisp
(require 'org)

(defun collapsible-template-start (id type post)
  """Template for inserting beginning of collapsible html element."""
  (concat "<p>
  <a data-toggle=\"collapse\" href=\"#collapsible-" id "\" role=\"button\" aria-expanded=\"false\" aria-controls=\"collapseExample\">
    Show " type "
  </a>
</p>
<div class=\"collapse\" id=\"collapsible-" id "\">
  <div class=\"card card-body\"> " post ))

(defvar collapsible-template-end "      </div>
    </div>
")

(defun my/filter-collapsible (text backend info)
  "Make Collapsible element."
  (pcase backend
    (`html
     (when (string-match-p (regexp-quote "collapsible") text)
       (concat (collapsible-template-start (org-id-new) "derivation" "") text collapsible-template-end)
       )
     )
    )
  )

(add-to-list 'org-export-filter-special-block-functions
	     #'my/filter-collapsible)
```

This elisp code wraps any block that contains the word "collapsible"
in html div's that makes the block collapsible, and adds a clickable
link with the text `Show derivation` before the block. Note that since
a unique id is required for the link to know which collapsible element
to work on, I used the `org-id-new` function to generate a random id.

To use this, I can add a block like the following to my `.org` file

```nil
#+begin_collapsible
  Schrödinger's equation
  \begin{align*}
      i \frac{d}{dt}\Psi = H\Psi
  \end{align*}
#+end_collapsible
```

Upon export it should look like this (click the link to expand the
collapsible, click again to collapse it)

{{< collapsible-split/start id=c1d62db02-ab72-4e76-b356-b207ac02821c type=derivation divclass="card card-body" >}}<div class="collapsible">
  <div></div>

Schrödinger's equation

\begin{align\*}
    i \frac{d}{dt}\Psi = H\Psi
\end{align\*}

</div>

{{< collapsible-split/end >}}It is also possible to nest collapsibles. A little care must be taken,
since org is not happy about nesting the same type of block. At each
nesting level some different superscript should be added to each
block, like below.

```nil
#+begin_collapsible
  Schrödinger's time dependent equation
  \begin{align*}
      i \frac{d}{dt}\Psi = H\Psi
  \end{align*}
#+begin_collapsible1
  Schrödinger's time independent equation
  \begin{align*}
      H\Psi & = E\Psi
  \end{align*}
#+end_collapsible1
#+end_collapsible
```

Which when exported will look like

{{< collapsible-split/start id=cec0a671c-f2b1-43d7-9f08-a0509ab084e6 type=derivation divclass="card card-body" >}}<div class="collapsible">
  <div></div>

Schrödinger's equation

\begin{align\*}
    i \frac{d}{dt}\Psi = H\Psi
\end{align\*}

{{< collapsible-split/start id=c7c84ae59-922d-4951-92fe-9188c6ef711b type=derivation divclass="card card-body" >}}<div class="collapsible1">
  <div></div>

Schrödinger's time independent equation

\begin{align\*}
    H\Psi & = E\Psi
\end{align\*}

</div>

{{< collapsible-split/end >}}

</div>

{{< collapsible-split/end >}}


## Hugo {#hugo}

Recently I started building my website using the static site generator
[Hugo](https://gohugo.io/). Using the [ox-hugo](https://ox-hugo.scripter.co/) exporter, I can export org files to
Hugo-compatible markdown. For me this provided a really nice workflow,
where I would use ox-hugo's auto export mode to automatically export
my org file on every file save, then have Hugo's livereload
automatically update a preview of the finished webpage. So now,
everytime I save my org file, I can almost instantly see the effect on
the rendered homepage, whereas before I had to run the export command
in emacs, and then shift to and refresh the browser.

Instead of exporting the html directly as in the above solution[^fn:1],
I decided to write a shortcode for this purpose. First I added the
following to my `~/.emacs` file

```lisp
(require 'org)

(defun my/filter-collapsible-replacer-shortcode (id type post)
  (concat "{{</* collapsible-split/start id=c" id " type=" type " divclass=\"card card-body\" */>}}"  post ))


(defun my/filter-collapsible-hugo (text backend info)
  "Make Collapsibles."
  (pcase backend
    (`hugo
     (when (string-match-p (regexp-quote "collapsible") text)
       (concat (my/filter-collapsible-replacer-shortcode (org-id-new) "derivation" "") text "{{</* collapsible-split/end */>}}")
       )
     )
    )
  )

(add-to-list 'org-export-filter-special-block-functions
	     #'my/filter-collapsible-hugo)
```

Next, I added the following to a file in
`[hugo-site-root]/layouts/shortcodes/collapsible-split/start.html`

```nil
<p>
  <a class="collapsed" data-toggle="collapse" href="#collapsible-{{ .Get "id" }}" role="button" aria-expanded="false" aria-controls="collapseExample">
    Show {{ .Get "type" }}
  </a>
</p>
<div class="collapse mathjax_ignore" id="collapsible-{{ .Get "id" }}">
  <div{{ if .Get "divclass" }} class="{{ .Get "divclass" }} collapsible"{{ else }}{{ end }}>
```

And in `[hugo-site-root]/layouts/shortcodes/collapsible-split/end.html` I added

```html
  </div>
</div>
```

I split the collapsible shortcode in two parts to make it work for
nested collapsibles. If I use something like `{{</* .Inner |
markdownify */>}}` or `{{</* .Inner */>}}` in a paired shortcode, then
either any shortcode inside would not be rendered or any markdown
inside would not be rendered.

[^fn:1]: The original solution would also have been complicated by html in markdown files does not get exported unless you set the goldmark unsafe option to true.
