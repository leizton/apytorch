#pragma once
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/tree.h>
#include <torch/csrc/jit/frontend/tree_views.h>
#include <memory>

namespace torch {
namespace jit {

struct Decl;
struct ParserImpl;
struct Lexer;

Decl mergeTypesFromTypeComment(
    const Decl& decl,
    const Decl& type_annotation_decl,
    bool is_method);

struct Parser {
  explicit Parser(const std::shared_ptr<Source>& src);
  TreeRef parseFunction(bool is_method);
  TreeRef parseClass();
  Decl parseTypeComment();
  Expr parseExp();
  Lexer& lexer();
  ~Parser();

 private:
  std::unique_ptr<ParserImpl> pImpl;
};

} // namespace jit
} // namespace torch
