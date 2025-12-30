-- Pandoc Lua filter to render callouts/admonitions nicely in LaTeX/PDF.
-- Converts fenced divs with callout classes into tcolorbox environments when
-- targeting LaTeX; leaves other output formats untouched.
--
-- Supported classes and their LaTeX environments:
--   .note      → CalloutNote (blue)
--   .tip       → TipBox (green)
--   .warning   → WarningBox (orange)
--   .danger    → DangerBox (red)
--   .info      → InfoBox (cyan)
--   .example   → ExampleBox (purple)
--   .important → ImportantBox (magenta)
--   .caution   → CautionBox (yellow)
--   .abstract  → AbstractBox (gray)
--   .summary   → AbstractBox (gray) - alias
--   .question  → QuestionBox (teal)
--   .quote     → QuoteBox (light gray)

-- Mapping from div class to LaTeX environment name and default title
local CALLOUT_MAP = {
  note = { env = "CalloutNote", default_title = "Note" },
  tip = { env = "TipBox", default_title = "Tip" },
  warning = { env = "WarningBox", default_title = "Warning" },
  danger = { env = "DangerBox", default_title = "Danger" },
  info = { env = "InfoBox", default_title = "Info" },
  example = { env = "ExampleBox", default_title = "Example" },
  important = { env = "ImportantBox", default_title = "Important" },
  caution = { env = "CautionBox", default_title = "Caution" },
  abstract = { env = "AbstractBox", default_title = "Abstract" },
  summary = { env = "AbstractBox", default_title = "Summary" },
  question = { env = "QuestionBox", default_title = "Question" },
  quote = { env = "QuoteBox", default_title = "Quote" },
}

local function get_callout_info(el)
  -- Check if element has any callout class and return the mapping
  for _, c in ipairs(el.classes) do
    if CALLOUT_MAP[c] then
      return CALLOUT_MAP[c], c
    end
  end
  return nil, nil
end

local function escape_latex(str)
  local replacements = {
    ['\\'] = '\\textbackslash{}',
    ['{'] = '\\{',
    ['}'] = '\\}',
    ['%'] = '\\%',
    ['$'] = '\\$',
    ['#'] = '\\#',
    ['&'] = '\\&',
    ['_'] = '\\_',
    ['^'] = '\\^{}',
    ['~'] = '\\~{}'
  }
  return str:gsub('[\\{}%%$#&_%^~]', function(c)
    return replacements[c] or c
  end)
end

local function wrap_in_callout(el, env, title)
  local blocks = {}
  table.insert(blocks, pandoc.RawBlock('latex', '\\begin{' .. env .. '}{' .. escape_latex(title) .. '}'))
  for _, block in ipairs(el.content) do
    table.insert(blocks, block)
  end
  table.insert(blocks, pandoc.RawBlock('latex', '\\end{' .. env .. '}'))
  return blocks
end

function Div(el)
  local callout_info, class_name = get_callout_info(el)

  if callout_info then
    local title = el.attributes['title'] or callout_info.default_title
    if FORMAT:match('latex') then
      return wrap_in_callout(el, callout_info.env, title)
    end
    -- Non-LaTeX outputs keep the div with its class for downstream styling.
    return el
  end

  return nil
end
