--[[
Cross-reference Lua filter for Pandoc

Converts custom cross-reference syntax to LaTeX labels and refs.

Anchor definitions (placed after equations, definitions, etc.):
    {#EQ-1.1}  → \label{EQ-1.1}   (injected into display math when adjacent)
    {#DEF-1.2} → \phantomsection\label{DEF-1.2}
    {#THM-1.3} → \phantomsection\label{THM-1.3}

Inline references:
    #EQ-1.1    → \eqref{EQ-1.1}   (equation refs use eqref)
    #DEF-1.2   → \hyperref[DEF-1.2]{1.2}  (others use hyperref + ID core)
    #THM-1.3   → \hyperref[THM-1.3]{1.3}

Bracket references:
    [EQ-1.1]   → \eqref{EQ-1.1}
    [DEF-1.2]  → \hyperref[DEF-1.2]{1.2}

Supported prefixes:
    EQ-   : Equations (uses \eqref for proper formatting)
    DEF-  : Definitions
    THM-  : Theorems
    REM-  : Remarks
    ASM-  : Assumptions
    WDEF- : Working Definitions

Usage:
    pandoc input.md --lua-filter=crossrefs.lua -o output.tex
--]]

-- Prefixes that should use \eqref (for equations)
local EQUATION_PREFIXES = {
    ["EQ"] = true,
}

-- Collected set of reference IDs that are defined in the current document.
-- Populated in Pandoc(doc) via a pre-scan so forward references resolve.
local DEFINED_IDS = {}

-- All valid prefixes for cross-references
local VALID_PREFIXES = {
    ["EQ"] = true,
    ["DEF"] = true,
    ["THM"] = true,
    ["PROP"] = true,   -- Propositions
    ["LEM"] = true,    -- Lemmas
    ["COR"] = true,    -- Corollaries
    ["ASSUMP"] = true, -- Assumptions
    ["REM"] = true,
    ["ASM"] = true,
    ["WDEF"] = true,
}

local function ref_core(id)
    -- Example: THM-2.3.4-RN -> 2.3.4 ; EQ-10.4-prime -> 10.4 ; EQ-C.1 -> C.1
    local core = id:match("^[A-Z]+%-([^%-]+)")
    return core or id
end

local function make_non_eq_ref(id)
    return "\\hyperref[" .. id .. "]{" .. ref_core(id) .. "}"
end

--- Check if a reference ID starts with an equation prefix
-- @param id string The reference ID (e.g., "EQ-1.1")
-- @return boolean True if this is an equation reference
local function is_equation_ref(id)
    local prefix = id:match("^([A-Z]+)%-")
    return prefix and EQUATION_PREFIXES[prefix]
end

--- Check if a string is a valid cross-reference ID
-- @param id string The potential reference ID
-- @return boolean True if this is a valid reference
local function is_valid_ref(id)
    local prefix = id:match("^([A-Z]+)%-")
    return prefix and VALID_PREFIXES[prefix]
end

--- Generate LaTeX reference command for an ID
-- @param id string The reference ID (e.g., "EQ-1.1")
-- @return string LaTeX command (\eqref{} or \ref{})
local function make_ref_command(id)
    if is_equation_ref(id) then
        return "\\eqref{" .. id .. "}"
    else
        return make_non_eq_ref(id)
    end
end

local function is_defined_in_doc(id)
    return DEFINED_IDS[id] == true
end

local function parse_anchor_token(text)
    local id = text:match("^%{#([A-Z]+%-[%w%.%-]*%w)%}$")
    if id and is_valid_ref(id) then
        return id, ""
    end

    local id_with_punct, punct = text:match("^%{#([A-Z]+%-[%w%.%-]*%w)%}([%.,;:!?])$")
    if id_with_punct and is_valid_ref(id_with_punct) then
        return id_with_punct, punct
    end

    return nil, nil
end

local function collect_defined_ids(doc)
    local defined = {}

    local function mark(id)
        if id and is_valid_ref(id) then
            defined[id] = true
        end
    end

    local function walk_inlines(inlines)
        for _, inline in ipairs(inlines or {}) do
            if inline.t == "Str" then
                local anchor_id = parse_anchor_token(inline.text)
                if anchor_id then
                    mark(anchor_id)
                end
            elseif inline.t == "Span" then
                if inline.identifier and inline.identifier ~= "" then
                    mark(inline.identifier)
                end
                walk_inlines(inline.content)
            elseif inline.t == "Link" then
                walk_inlines(inline.content)
            end
        end
    end

    local function walk_blocks(blocks)
        for _, block in ipairs(blocks or {}) do
            if block.t == "Para" or block.t == "Plain" then
                walk_inlines(block.content)
            elseif block.t == "Header" then
                if block.identifier and block.identifier ~= "" then
                    mark(block.identifier)
                end
                walk_inlines(block.content)
            elseif block.t == "Div" then
                if block.identifier and block.identifier ~= "" then
                    mark(block.identifier)
                end
                walk_blocks(block.content)
            elseif block.t == "BlockQuote" then
                walk_blocks(block.content)
            elseif block.t == "BulletList" or block.t == "OrderedList" then
                local items = block.content
                if block.t == "OrderedList" then
                    items = block.content[2]
                end
                for _, item in ipairs(items or {}) do
                    walk_blocks(item)
                end
            elseif block.t == "Table" then
                -- Pandoc 3.x Table has nested structures; best-effort scan of all cells.
                local head = block.head
                if head and head.rows then
                    for _, row in ipairs(head.rows) do
                        for _, cell in ipairs(row.cells or {}) do
                            walk_blocks(cell.contents)
                        end
                    end
                end
                local bodies = block.bodies or {}
                for _, body in ipairs(bodies) do
                    for _, row in ipairs(body.body or {}) do
                        for _, cell in ipairs(row.cells or {}) do
                            walk_blocks(cell.contents)
                        end
                    end
                end
                local foot = block.foot
                if foot and foot.rows then
                    for _, row in ipairs(foot.rows) do
                        for _, cell in ipairs(row.cells or {}) do
                            walk_blocks(cell.contents)
                        end
                    end
                end
            end
        end
    end

    walk_blocks(doc.blocks)
    return defined
end

--- Process Para elements to:
-- 1) attach equation labels inside display math when the next inline is {#EQ-*};
-- 2) convert trailing {#DEF-*}, {#THM-*}, ... anchor tokens into \phantomsection\label{...}.
local function process_inlines_block(el)
    if not FORMAT:match("latex") then
        return nil
    end

    local new_inlines = pandoc.List()
    local i = 1
    while i <= #el.content do
        local inline = el.content[i]

        if inline.t == "Math" and inline.mathtype == "DisplayMath" then
            -- Pattern: DisplayMath, SoftBreak, "{#EQ-...}"
            local sb = el.content[i + 1]
            local tok = el.content[i + 2]
            if sb and sb.t == "SoftBreak" and tok and tok.t == "Str" then
                local anchor_id = parse_anchor_token(tok.text)
                if anchor_id and is_equation_ref(anchor_id) then
                    local math_text = inline.text
                    if not math_text:find("\\label%{") and not math_text:find("\\label%[") and not math_text:find("\\label%s*%{") then
                        math_text = math_text .. "\\label{" .. anchor_id .. "}"
                    end
                    new_inlines:insert(pandoc.Math(inline.mathtype, math_text))
                    i = i + 3
                    goto continue
                end
            end
        end

        if inline.t == "Str" then
            local anchor_id, anchor_punct = parse_anchor_token(inline.text)
            if anchor_id then
                -- For non-equation anchors we want a stable hyperlink target (not a counter-based number).
                new_inlines:insert(pandoc.RawInline("latex", "\\phantomsection\\label{" .. anchor_id .. "}"))
                if anchor_punct and anchor_punct ~= "" then
                    new_inlines:insert(pandoc.Str(anchor_punct))
                end
                i = i + 1
                goto continue
            end
        end

        new_inlines:insert(inline)
        i = i + 1
        ::continue::
    end

    el.content = new_inlines
    return el
end

function Para(el)
    return process_inlines_block(el)
end

-- Pandoc often represents tight list items as Plain blocks (not Para). We handle
-- them identically so equation anchors like "{#EQ-...}" are still attached.
function Plain(el)
    return process_inlines_block(el)
end

--- Process Span elements (handles {#ID} anchor syntax when Pandoc attaches identifiers)
function Span(el)
    -- Only process when targeting LaTeX
    if not FORMAT:match("latex") then
        return nil
    end

    -- Check if this span has an identifier that matches our pattern
    if el.identifier and el.identifier ~= "" then
        if is_valid_ref(el.identifier) then
            -- This is an anchor definition - emit \label{}
            -- Keep any content inside the span, add label after
            local label = pandoc.RawInline("latex", "\\phantomsection\\label{" .. el.identifier .. "}")

            if #el.content == 0 then
                -- Empty span, just return the label
                return label
            else
                -- Span has content, append label to content
                local result = pandoc.List(el.content)
                result:insert(label)
                return result
            end
        end
    end

    return nil
end

--- Process Str elements (handles #ID and [ID] reference syntax)
function Str(el)
    -- Only process when targeting LaTeX
    if not FORMAT:match("latex") then
        return nil
    end

    local text = el.text

    -- Pattern 1: Hash reference #TYPE-X.Y or #TYPE-X.Y.Z or #TYPE-X.Y-suffix
    -- Must end with alphanumeric (not a dot - that's sentence punctuation)
    local hash_ref = text:match("^#([A-Z]+%-[%w%.%-]*%w)(.?)$")
    if hash_ref and is_valid_ref(hash_ref) then
        local trailing = text:match("^#[A-Z]+%-[%w%.%-]*%w(.)$") or ""
        if trailing == "" then
            if is_defined_in_doc(hash_ref) then
                return pandoc.RawInline("latex", make_ref_command(hash_ref))
            end
            return pandoc.Str(hash_ref)
        else
            -- There's trailing punctuation (like a period) - keep it
            if not is_defined_in_doc(hash_ref) then
                return {
                    pandoc.Str(hash_ref),
                    pandoc.Str(trailing),
                }
            end
            return {
                pandoc.RawInline("latex", make_ref_command(hash_ref)),
                pandoc.Str(trailing)
            }
        end
    end

    -- Also handle case where ref is followed by punctuation in same token
    local ref_with_punct, punct = text:match("^#([A-Z]+%-[%w%.%-]*%w)([%.,;:!?])$")
    if ref_with_punct and is_valid_ref(ref_with_punct) then
        if not is_defined_in_doc(ref_with_punct) then
            return {
                pandoc.Str(ref_with_punct),
                pandoc.Str(punct),
            }
        end
        return {
            pandoc.RawInline("latex", make_ref_command(ref_with_punct)),
            pandoc.Str(punct)
        }
    end

    -- Pattern 2: Bracket reference [TYPE-X.Y] (as a single Str token)
    -- Note: Pandoc usually breaks [text] into Link, but sometimes keeps as Str
    local bracket_ref = text:match("^%[([A-Z]+%-[%w%.%-]*%w)%]$")
    if bracket_ref and is_valid_ref(bracket_ref) then
        if not is_defined_in_doc(bracket_ref) then
            return pandoc.Str(bracket_ref)
        end
        return pandoc.RawInline("latex", make_ref_command(bracket_ref))
    end

    -- Bracket reference followed by punctuation in the same token: [TYPE-X.Y].
    local bracket_with_punct, bracket_punct = text:match("^%[([A-Z]+%-[%w%.%-]*%w)%]([%.,;:!?])$")
    if bracket_with_punct and is_valid_ref(bracket_with_punct) then
        if not is_defined_in_doc(bracket_with_punct) then
            return {
                pandoc.Str(bracket_with_punct),
                pandoc.Str(bracket_punct),
            }
        end
        return {
            pandoc.RawInline("latex", make_ref_command(bracket_with_punct)),
            pandoc.Str(bracket_punct),
        }
    end

    return nil
end

--- Process Link elements (handles [ID] that Pandoc parses as links)
-- When Pandoc sees [EQ-1.1] it may parse it as a link with empty target
function Link(el)
    -- Only process when targeting LaTeX
    if not FORMAT:match("latex") then
        return nil
    end

    -- Check if this is a reference-style "link" with no URL
    -- [EQ-1.1] becomes a Link with target "" or "#EQ-1.1"
    if el.target == "" or el.target:match("^#") then
        -- Extract the link text
        local link_text = pandoc.utils.stringify(el.content)

        -- Check if it matches our reference pattern
        if is_valid_ref(link_text) then
            if not is_defined_in_doc(link_text) then
                return pandoc.Str(link_text)
            end
            return pandoc.RawInline("latex", make_ref_command(link_text))
        end

        -- Also check target (for [text](#EQ-1.1) style)
        if el.target:match("^#") then
            local target_ref = el.target:sub(2)  -- Remove leading #
            if is_valid_ref(target_ref) then
                if not is_defined_in_doc(target_ref) then
                    return pandoc.Str(target_ref)
                end
                return pandoc.RawInline("latex", make_ref_command(target_ref))
            end
        end
    end

    return nil
end

--- Process Div elements to find anchors in header attributes
-- Headers like "## Section {#DEF-1.1}" get the ID in the Div/Header
function Header(el)
    -- Only process when targeting LaTeX
    if not FORMAT:match("latex") then
        return nil
    end

    -- Pandoc already emits \label{<identifier>} for headers with identifiers in the
    -- LaTeX writer. We deliberately avoid adding another label here to prevent
    -- multiply-defined label warnings and to keep section titles (moving arguments)
    -- free of injected raw LaTeX.
    return nil
end

function Pandoc(doc)
    if not FORMAT:match("latex") then
        return doc
    end
    DEFINED_IDS = collect_defined_ids(doc)
    return doc
end

-- Return the filter functions
return {
    { Pandoc = Pandoc },
    { Para = Para },
    { Plain = Plain },
    { Span = Span },
    { Str = Str },
    { Link = Link },
    { Header = Header },
}
