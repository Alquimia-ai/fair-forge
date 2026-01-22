# Mintlify Quick Reference

When you need detailed information about Mintlify components or configuration, fetch:

```
https://www.mintlify.com/docs/llms.txt
```

## Common Components

### CodeGroup

Multiple code examples with tabs:

```mdx
<CodeGroup>
```python Python
print("Hello")
```

```javascript JavaScript
console.log("Hello")
```
</CodeGroup>
```

### Tabs

Organize content by category:

```mdx
<Tabs>
  <Tab title="Option 1">
    Content for option 1
  </Tab>
  <Tab title="Option 2">
    Content for option 2
  </Tab>
</Tabs>
```

### Cards

Navigation or highlight cards:

```mdx
<CardGroup cols={2}>
  <Card title="Title" icon="icon-name" href="/path">
    Description text
  </Card>
</CardGroup>
```

### Accordions

Collapsible sections:

```mdx
<AccordionGroup>
  <Accordion title="Question 1">
    Answer content
  </Accordion>
</AccordionGroup>
```

### Callouts

Notes and warnings:

```mdx
<Note>
  Important information
</Note>

<Warning>
  Critical warning
</Warning>

<Info>
  Helpful tip
</Info>
```

### Steps

Numbered instructions:

```mdx
<Steps>
  <Step title="First Step">
    Do this first
  </Step>
  <Step title="Second Step">
    Then do this
  </Step>
</Steps>
```

## Frontmatter

Every MDX file needs frontmatter:

```yaml
---
title: Page Title
description: Short description for SEO
---
```

## docs.json Navigation

Structure for navigation in docs.json:

```json
{
  "navigation": {
    "tabs": [
      {
        "tab": "Tab Name",
        "groups": [
          {
            "group": "Group Name",
            "pages": [
              "folder/page-name"
            ]
          }
        ]
      }
    ]
  }
}
```

## Icons

Common icons from Font Awesome:
- `rocket` - Getting started
- `gear` - Configuration
- `code` - Code/API
- `book` - Documentation
- `chart-pie` - Metrics/Analytics
- `play` - Run/Execute
- `database` - Storage
- `cloud` - Cloud services
- `robot` - AI/Automation
- `bolt` - Performance
- `check` - Success/Validation
- `heart` - Humanity/Emotion
- `shield` - Security/Bias
- `comments` - Conversation
- `file-import` - Import/Load
- `sitemap` - Structure/Strategy

## For More Details

Always fetch `https://www.mintlify.com/docs/llms.txt` for:
- Complete component documentation
- Advanced configuration options
- Styling and theming
- API documentation features
- Custom components
