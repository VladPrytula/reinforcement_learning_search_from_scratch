-- Pandoc Lua filter to render note-style callouts nicely in LaTeX/PDF.
-- Converts fenced divs with class "note" into a tcolorbox environment when
-- targeting LaTeX; leaves other output formats untouched.

local function has_class(el, class)
  for _, c in ipairs(el.classes) do
    if c == class then
      return true
    end
  end
  return false
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
  if has_class(el, 'note') then
    local title = el.attributes['title'] or 'Note'
    if FORMAT:match('latex') then
      return wrap_in_callout(el, 'CalloutNote', title)
    end
    -- Non-LaTeX outputs keep the div with its class for downstream styling.
    return el
  end
  return nil
end
