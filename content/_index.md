---
title: 'Home'
date: 2023-10-24
type: landing

design:
  # Default section spacing
  spacing: "3.5rem"

sections:
  - block: hero
    content:
      title: KVCache.ai
      text: "A collaborative endeavor with leading industry partners such as Approaching.AI and Moonshot AI. The project focuses on developing effective and practical techniques that enrich both academic research and open-source development."
      icon: icon.png
      # primary_action:
      #   text: Get Started
      #   url: https://github.com/kvcache-ai
      #   icon: rocket-launch
      # secondary_action:
      #   text: Read the docs
      #   url: /docs/
      announcement:
        text: "The project's newest blog! "
        link:
          text: "Read more"
          url: "/blog/"
    design:
      spacing:
        padding: [0, 0, 0, 0]
        margin: [0, 0, 0, 0]
      # For full-screen, add `min-h-screen` below
      css_class: ""
      background:
        color: ""
        image:
          # Add your image background to `assets/media/`.
          filename: ""
          filters:
            brightness: 0.5

  - block: top-blog
    id: top post
    content:
      title: ''
      sort_by: 'home_weight'
      filters:
        folders: 
          - blog
          - repo
    design:
      view: top-blog-view

  # - block: collection
  #   id: githubs
  #   content:
  #     title: Github repo
  #     filters:
  #       folders: 
  #         - repo
  #   design:
  #     view: card

  # - block: collection
  #   id: blogs
  #   content:
  #     title: ''
  #     count: 6
  #     filters:
  #       folders: 
  #         - blog
  #   design:
  #     view: article-grid


  

  # - block: features
  #   id: features
  #   content:
  #     title: Features
  #     text: Collaborate, publish, and maintain technical knowledge with an all-in-one documentation site. Used by 100,000+ startups, enterprises, and researchers.
  #     items:
  #       - name: Optimized SEO
  #         icon: magnifying-glass
  #         description: Automatic sitemaps, RSS feeds, and rich metadata take the pain out of SEO and syndication.
  #       - name: Fast
  #         icon: bolt
  #         description: Super fast page load with Tailwind CSS and super fast site building with Hugo.
  #       - name: Easy
  #         icon: sparkles
  #         description: One-click deployment to GitHub Pages. Have your new website live within 5 minutes!
  #       - name: No-Code
  #         icon: code-bracket
  #         description: Edit and design your site just using rich text (Markdown) and configurable YAML parameters.
  #       - name: Highly Rated
  #         icon: star
  #         description: Rated 5-stars by the community.
  #       - name: Swappable Blocks
  #         icon: rectangle-group
  #         description: Build your pages with blocks - no coding required!
  # - block: cta-card
  #   content:
  #     title: "Start Writing with the #1 Effortless Documentation Platform"
  #     text: Hugo Blox Docs Theme brings all your technical knowledge together in a single, centralized knowledge base. Easily search and edit it with the tools you use every day!
  #     button:
  #       text: Get Started
  #       url: https://hugoblox.com/templates/details/docs/
  #   design:
  #     card:
  #       # Card background color (CSS class)
  #       css_class: "bg-primary-700"
  #       css_style: ""
---
